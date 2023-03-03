from __future__ import absolute_import, division, print_function

import os
import zipfile
import logging
import sys
from pathlib import Path
import yaml
import json
import numpy as np
import torch
import click
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP

from sedna.common.class_factory import ClassType, ClassFactory
from sedna.common.config import Context
from sedna.common.file_ops import FileOps

from segm.utils import distributed
import segm.utils.torch as ptu
from segm import config

from segm.model.factory import create_segmenter
from segm.optim.factory import create_optimizer, create_scheduler
from segm.model.utils import num_params

from custom_data.factory import create_dataset

from timm.utils import NativeScaler
from contextlib import suppress

from segm.utils.distributed import sync_model
from segm.engine import train_one_epoch, evaluate


dataset_kwargs={}
algorithm_kwargs={}
optimizer_kwargs={}
net_kwargs={}
inference_kwargs={}

logging.disable(logging.WARNING)

__all__ = ["BaseModel"]

os.environ['BACKEND_TYPE'] = 'TORCH'


@ClassFactory.register(ClassType.GENERAL, alias="ViT")
class BaseModel:

    def __init__(self, batch_size=8, amp=False, resume=True, **kwargs):
        """
        """
        algorithm_kwargs["num_epochs"] = kwargs.get("num_epochs", 216)
        algorithm_kwargs["start_epoch"] = kwargs.get("start_epoch", 214)
        optimizer_kwargs["lr"] = kwargs.get("learning_rate", 0.01)
        optimizer_kwargs["momentum"] = kwargs.get("momentum", 0.9)

        # start distributed mode
        ptu.set_gpu_mode(True)
        distributed.init_process()

        if batch_size:
            world_batch_size = batch_size

        # experiment config
        batch_size = world_batch_size // ptu.world_size
        dataset_kwargs['batch_size'] = batch_size
        algorithm_kwargs['batch_size'] = batch_size

        os.environ["MODEL_NAME"] = "model_vit.pth"

        self.checkpoint_path = self.load(Context.get_parameters("base_model_url"))
        self.log_dir, model_name = os.path.split(self.checkpoint_path)

        self.variant = dict(
            world_batch_size=world_batch_size,
            version="normal",
            resume=resume,
            dataset_kwargs=dataset_kwargs,
            algorithm_kwargs=algorithm_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            net_kwargs=net_kwargs,
            amp=amp,
            log_dir=self.log_dir,
            inference_kwargs=inference_kwargs,
        )

    def train(self, train_data, valid_data=None, **kwargs):
        # dataset
        dataset_kwargs = variant["dataset_kwargs"]
        dataset_dir, train_index = os.path.split(train_data)
        train_loader = create_dataset(dataset_dir, dataset_kwargs)
        val_kwargs = dataset_kwargs.copy()
        val_kwargs["split"] = "val"
        val_kwargs["batch_size"] = 1
        val_kwargs["crop"] = False
        val_loader = create_dataset(dataset_dir, val_kwargs)
        n_cls = train_loader.unwrapped.n_cls
    
        # model
        net_kwargs = variant["net_kwargs"]
        net_kwargs["n_cls"] = n_cls
        model = create_segmenter(net_kwargs)
        model.to(ptu.device)
    
        # optimizer
        optimizer_kwargs = variant["optimizer_kwargs"]
        optimizer_kwargs["iter_max"] = len(train_loader) * optimizer_kwargs["epochs"]
        optimizer_kwargs["iter_warmup"] = 0.0
        opt_args = argparse.Namespace()
        opt_vars = vars(opt_args)
        for k, v in optimizer_kwargs.items():
            opt_vars[k] = v
        optimizer = create_optimizer(opt_args, model)
        lr_scheduler = create_scheduler(opt_args, optimizer)
        num_iterations = 0
        amp_autocast = suppress
        loss_scaler = None
        if amp:
            amp_autocast = torch.cuda.amp.autocast
            loss_scaler = NativeScaler()
    
        # resume
        if resume and self.checkpoint_path.exists():
            print(f"Resuming training from checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if loss_scaler and "loss_scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["loss_scaler"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            variant["algorithm_kwargs"]["start_epoch"] = checkpoint["epoch"] + 1
        else:
            sync_model(self.log_dir, model)
    
        if ptu.distributed:
            model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)
    
        # save config
        variant_str = yaml.dump(variant)
        print(f"Configuration:\n{variant_str}")
        variant["net_kwargs"] = net_kwargs
        variant["dataset_kwargs"] = dataset_kwargs
        # log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / "variant.yml", "w") as f:
            f.write(variant_str)
    
        # train
        start_epoch = variant["algorithm_kwargs"]["start_epoch"]
        num_epochs = variant["algorithm_kwargs"]["num_epochs"]
        eval_freq = variant["algorithm_kwargs"]["eval_freq"]
    
        model_without_ddp = model
        if hasattr(model, "module"):
            model_without_ddp = model.module
    
        val_seg_gt = val_loader.dataset.get_gt_seg_maps()
    
        print(f"Train dataset length: {len(train_loader.dataset)}")
        print(f"Val dataset length: {len(val_loader.dataset)}")
        print(f"Encoder parameters: {num_params(model_without_ddp.encoder)}")
        print(f"Decoder parameters: {num_params(model_without_ddp.decoder)}")
    
        for epoch in range(start_epoch, num_epochs):
            # train for one epoch
            train_logger = train_one_epoch(
                model,
                train_loader,
                optimizer,
                lr_scheduler,
                epoch,
                amp_autocast,
                loss_scaler,
            )
    
            # save checkpoint
            if ptu.dist_rank == 0:
                snapshot = dict(
                    model=model_without_ddp.state_dict(),
                    optimizer=optimizer.state_dict(),
                    n_cls=model_without_ddp.n_cls,
                    lr_scheduler=lr_scheduler.state_dict(),
                )
                if loss_scaler is not None:
                    snapshot["loss_scaler"] = loss_scaler.state_dict()
                snapshot["epoch"] = epoch
                torch.save(snapshot, self.checkpoint_path)
    
            # evaluate
            eval_epoch = epoch % eval_freq == 0 or epoch == num_epochs - 1
            if eval_epoch:
                eval_logger = evaluate(
                    model,
                    val_loader,
                    val_seg_gt,
                    window_size,
                    window_stride,
                    amp_autocast,
                )
                print(f"Stats [{epoch}]:", eval_logger, flush=True)
                print("")
    
            # log stats
            if ptu.dist_rank == 0:
                train_stats = {
                    k: meter.global_avg for k, meter in train_logger.meters.items()
                }
                val_stats = {}
                if eval_epoch:
                    val_stats = {
                        k: meter.global_avg for k, meter in eval_logger.meters.items()
                    }
    
                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"val_{k}": v for k, v in val_stats.items()},
                    "epoch": epoch,
                    "num_updates": (epoch + 1) * len(train_loader),
                }
    
                with open(self.log_dir / "log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
    
        distributed.barrier()
        distributed.destroy_process()
        return self.checkpoint_path

    def save(self, model_path):
        if not model_path:
            raise Exception("model path is None.")

        model_dir, model_name = os.path.split(self.checkpoint_path)
        models = [model for model in os.listdir(
            model_dir) if model_name in model]

        if os.path.splitext(model_path)[-1] != ".zip":
            model_path = os.path.join(model_path, "model.zip")

        if not os.path.isdir(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        with zipfile.ZipFile(model_path, "w") as f:
            for model_file in models:
                model_file_path = os.path.join(model_dir, model_file)
                f.write(model_file_path, model_file,
                        compress_type=zipfile.ZIP_DEFLATED)

        return model_path

    def predict(self, data, **kwargs):
        # load test data
        return self.validator.validate() # y_actual

    def load(self, model_url): # extension may vary
        # model_url = '/home/wxc/dev/ianvs/models/model_best_vit.pth'
        if FileOps.exists(model_url):
            self.validator.new_state_dict = torch.load(
                model_url, map_location=torch.device("cpu"))
        else:
            raise Exception("model url does not exist.")
        self.validator.model = load_my_state_dict(
            self.validator.model, self.validator.new_state_dict['state_dict'])
        return model_url
    
    def evaluate(self, data, model_path, **kwargs): # fix this
        if data is None or data.x is None or data.y is None:
            raise Exception("Prediction data is None")

        self.load(model_path)
        predict_dict = self.predict(data.x)
        metric_name, metric_func = kwargs.get("metric")
        if callable(metric_func):
            return {"f1_score": metric_func(data.y, predict_dict)}
        else:
            raise Exception(f"not found model metric func(name={metric_name}) in model eval phase")

    def _preprocess(self, image_urls):
        # maybe

        return transformed_images


###############################################################################################


def main(
    log_dir,
    batch_size,
    amp,
    resume,
):
    # start distributed mode
    ptu.set_gpu_mode(True)
    distributed.init_process()
    
    if batch_size:
        world_batch_size = batch_size

    # experiment config
    batch_size = world_batch_size // ptu.world_size
    dataset_kwargs['batch_size'] = batch_size
    algorithm_kwargs['batch_size'] = batch_size
    
    variant = dict(
        world_batch_size=world_batch_size,
        version="normal",
        resume=resume,
        dataset_kwargs= #config,
        algorithm_kwargs= #config,
        optimizer_kwargs= #config,
        net_kwargs= #config,
        amp=amp,
        log_dir=log_dir,
        inference_kwargs= #config,
    )

    # dataset
    dataset_kwargs = variant["dataset_kwargs"]

    train_loader = create_dataset(dataset_kwargs)
    val_kwargs = dataset_kwargs.copy()
    val_kwargs["split"] = "val"
    val_kwargs["batch_size"] = 1
    val_kwargs["crop"] = False
    val_loader = create_dataset(val_kwargs)
    n_cls = train_loader.unwrapped.n_cls

    # model
    net_kwargs = variant["net_kwargs"]
    net_kwargs["n_cls"] = n_cls
    model = create_segmenter(net_kwargs)
    model.to(ptu.device)

    # optimizer
    optimizer_kwargs = variant["optimizer_kwargs"]
    optimizer_kwargs["iter_max"] = len(train_loader) * optimizer_kwargs["epochs"]
    optimizer_kwargs["iter_warmup"] = 0.0
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v
    optimizer = create_optimizer(opt_args, model)
    lr_scheduler = create_scheduler(opt_args, optimizer)
    num_iterations = 0
    amp_autocast = suppress
    loss_scaler = None
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # resume
    if resume and checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if loss_scaler and "loss_scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["loss_scaler"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        variant["algorithm_kwargs"]["start_epoch"] = checkpoint["epoch"] + 1
    else:
        sync_model(log_dir, model)

    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    # save config
    variant_str = yaml.dump(variant)
    print(f"Configuration:\n{variant_str}")
    variant["net_kwargs"] = net_kwargs
    variant["dataset_kwargs"] = dataset_kwargs
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "variant.yml", "w") as f:
        f.write(variant_str)

    # train
    start_epoch = variant["algorithm_kwargs"]["start_epoch"]
    num_epochs = variant["algorithm_kwargs"]["num_epochs"]
    eval_freq = variant["algorithm_kwargs"]["eval_freq"]

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    val_seg_gt = val_loader.dataset.get_gt_seg_maps()

    print(f"Train dataset length: {len(train_loader.dataset)}")
    print(f"Val dataset length: {len(val_loader.dataset)}")
    print(f"Encoder parameters: {num_params(model_without_ddp.encoder)}")
    print(f"Decoder parameters: {num_params(model_without_ddp.decoder)}")

    for epoch in range(start_epoch, num_epochs):
        # train for one epoch
        train_logger = train_one_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch,
            amp_autocast,
            loss_scaler,
        )

        # save checkpoint
        if ptu.dist_rank == 0:
            snapshot = dict(
                model=model_without_ddp.state_dict(),
                optimizer=optimizer.state_dict(),
                n_cls=model_without_ddp.n_cls,
                lr_scheduler=lr_scheduler.state_dict(),
            )
            if loss_scaler is not None:
                snapshot["loss_scaler"] = loss_scaler.state_dict()
            snapshot["epoch"] = epoch
            torch.save(snapshot, checkpoint_path)

        # evaluate
        eval_epoch = epoch % eval_freq == 0 or epoch == num_epochs - 1
        if eval_epoch:
            eval_logger = evaluate(
                model,
                val_loader,
                val_seg_gt,
                window_size,
                window_stride,
                amp_autocast,
            )
            print(f"Stats [{epoch}]:", eval_logger, flush=True)
            print("")

        # log stats
        if ptu.dist_rank == 0:
            train_stats = {
                k: meter.global_avg for k, meter in train_logger.meters.items()
            }
            val_stats = {}
            if eval_epoch:
                val_stats = {
                    k: meter.global_avg for k, meter in eval_logger.meters.items()
                }

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "num_updates": (epoch + 1) * len(train_loader),
            }

            with open(log_dir / "log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    distributed.barrier()
    distributed.destroy_process()
    sys.exit(1)


if __name__ == "__main__":
    main()
