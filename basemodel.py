from __future__ import absolute_import, division, print_function

import os
import sys
import zipfile
import logging
from pathlib import Path
import yaml
import json
import torch
import argparse
import shutil
from torch.nn.parallel import DistributedDataParallel as DDP

from sedna.common.class_factory import ClassType, ClassFactory
from sedna.common.config import Context

from segm.utils import distributed
import segm.utils.torch as ptu
from segm.utils.logger import MetricLogger

from segm.model.factory import create_segmenter
from segm.optim.factory import create_optimizer, create_scheduler
from segm.model.utils import num_params

from custom_data.factory import create_dataset
from custom_data.cityscapes_configs import algorithm_kwargs, dataset_kwargs, inference_kwargs, net_kwargs, optimizer_kwargs

from timm.utils import NativeScaler
from contextlib import suppress

from segm.utils.distributed import sync_model
from segm.engine import train_one_epoch, evaluate
from segm.eval.miou import process_batch, save_im
from segm.metrics import gather_data


logging.disable(logging.WARNING)

__all__ = ["BaseModel"]

os.environ['BACKEND_TYPE'] = 'TORCH'
os.environ["RESULT_SAVED_URL"] = './singletask_learning_bench/workspace' # only for testing purposes


@ClassFactory.register(ClassType.GENERAL, alias="ViT")
class BaseModel:

    def __init__(self, batch_size=8, amp=False, resume=False, **kwargs): # resume giving error (so false for now)
        """
        """
        algorithm_kwargs["num_epochs"] = kwargs.get("num_epochs", 216)
        algorithm_kwargs["start_epoch"] = kwargs.get("start_epoch", 214)
        optimizer_kwargs["lr"] = kwargs.get("learning_rate", 0.01)
        optimizer_kwargs["momentum"] = kwargs.get("momentum", 0.9)

        # start distributed mode
        ptu.set_gpu_mode(True) # remove from comment after testing
        distributed.init_process()

        if batch_size:
            world_batch_size = batch_size

        # experiment config
        batch_size = world_batch_size // ptu.world_size
        dataset_kwargs['batch_size'] = batch_size
        algorithm_kwargs['batch_size'] = batch_size

        self.log_dir = Path(os.getenv("RESULT_SAVED_URL"))
        print("type log", self.log_dir, type(self.log_dir))
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

        os.environ["MODEL_NAME"] = "model.zip"
        print("base_model:", Context.get_parameters("base_model_url")) #for testing
        self.checkpoint_path = Path(self.load(Context.get_parameters("base_model_url")))
        print("chkpt:", self.checkpoint_path) #for testing

    def train(self, train_data, valid_data=None, **kwargs):
        # dataset
        dataset_kwargs = self.variant["dataset_kwargs"]
        dataset_dir = Path(train_data.x[0].split("/image_file_index")[0]) # temporary solution
        print(dataset_dir)

        train_loader = create_dataset(dataset_dir, dataset_kwargs)
        val_kwargs = dataset_kwargs.copy()
        val_kwargs["split"] = "val"
        val_kwargs["batch_size"] = 1
        val_kwargs["crop"] = False
        val_loader = create_dataset(dataset_dir, val_kwargs)
        n_cls = train_loader.unwrapped.n_cls

        # model
        net_kwargs = self.variant["net_kwargs"]
        net_kwargs["n_cls"] = n_cls
        model = create_segmenter(net_kwargs)
        model.to(ptu.device)

        # optimizer
        optimizer_kwargs = self.variant["optimizer_kwargs"]
        optimizer_kwargs["iter_max"] = len(train_loader) * optimizer_kwargs["epochs"]
        optimizer_kwargs["iter_warmup"] = 0.0
        opt_args = argparse.Namespace()
        opt_vars = vars(opt_args)
        for k, v in optimizer_kwargs.items():
            opt_vars[k] = v
        optimizer = create_optimizer(opt_args, model)
        lr_scheduler = create_scheduler(opt_args, optimizer)
        # num_iterations = 0
        amp_autocast = suppress
        loss_scaler = None
        amp = self.variant["amp"]
        if amp:
            amp_autocast = torch.cuda.amp.autocast
            loss_scaler = NativeScaler()

        # resume
        resume = self.variant["resume"]
        if resume and self.checkpoint_path.exists():
            print(f"Resuming training from checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if loss_scaler and "loss_scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["loss_scaler"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.variant["algorithm_kwargs"]["start_epoch"] = checkpoint["epoch"] + 1
        else:
            sync_model(self.log_dir, model)

        if ptu.distributed:
            model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

        # save config
        variant_str = yaml.dump(self.variant)
        print(f"Configuration:\n{variant_str}")
        self.variant["net_kwargs"] = net_kwargs
        self.variant["dataset_kwargs"] = dataset_kwargs
        # log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / "variant.yml", "w") as f:
            f.write(variant_str)

        # train
        start_epoch = self.variant["algorithm_kwargs"]["start_epoch"]
        num_epochs = self.variant["algorithm_kwargs"]["num_epochs"]
        eval_freq = self.variant["algorithm_kwargs"]["eval_freq"]

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
            window_size = self.variant["inference_kwargs"]['window_size']
            window_stride = self.variant["inference_kwargs"]['window_stride']
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
        model_path = "./initial_model/model.zip" # maybe some error faced by others as well
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

    def predict(self, data_loader, save_images=False, **kwargs):
        inference_output_dir = Path(os.getenv("RESULT_SAVED_URL"))

        dataset_name = self.variant["dataset_kwargs"]["dataset"]
        logger = MetricLogger(delimiter="  ")
        header = ""
        print_freq = 50
        window_size = self.variant["inference_kwargs"]['window_size']
        window_stride = self.variant["inference_kwargs"]['window_stride']
        window_batch_size = 4
        normalization = data_loader.dataset.normalization
        blend = True

        ims = {}
        seg_pred_maps = {}
        idx = 0
        for batch in logger.log_every(data_loader, print_freq, header):
            colors = batch["colors"]
            filename, im, seg_pred = process_batch(
                self.test_model, batch, window_size, window_stride, window_batch_size,
            )
            ims[filename] = im
            seg_pred_maps[filename] = seg_pred
            idx += 1

        seg_gt_maps = data_loader.dataset.get_gt_seg_maps()

        if save_images:
            save_dir = inference_output_dir / "images"
            if ptu.dist_rank == 0:
                if save_dir.exists():
                    shutil.rmtree(save_dir)
                save_dir.mkdir()
            if ptu.distributed:
                torch.distributed.barrier()

            for name in sorted(ims):
                instance_dir = save_dir
                filename = name

                if dataset_name == "cityscapes":
                    filename_list = name.split("/")
                    instance_dir = instance_dir / filename_list[0]
                    filename = filename_list[-1]
                    if not instance_dir.exists():
                        instance_dir.mkdir()

                save_im(
                    instance_dir,
                    filename,
                    ims[name],
                    seg_pred_maps[name],
                    torch.tensor(seg_gt_maps[name]),
                    colors,
                    blend,
                    normalization,
                )
            if ptu.dist_rank == 0:
                shutil.make_archive(save_dir, "zip", save_dir)
                # shutil.rmtree(save_dir)
                print(f"Saved eval images in {save_dir}.zip")

        if ptu.distributed:
            torch.distributed.barrier()
            seg_pred_maps = gather_data(seg_pred_maps)

        return seg_pred_maps

    def load(self, model_path):
        model_path = "./initial_model/model.zip" # maybe some error faced by others as well
        # load checkpoint path
        if model_path:
            model_dir = os.path.split(model_path)[0]
            with zipfile.ZipFile(model_path, "r") as f:
                f.extractall(path=model_dir)
                ckpt_name = os.path.basename(f.namelist()[0])
                index = ckpt_name.find("pth")
                ckpt_name = ckpt_name[:index + 4]
            self.checkpoint_path = os.path.join(model_dir, ckpt_name)

        else:
            raise Exception(f"model path [{model_path}] is None.")

        # load test model
        checkpoint = torch.load(self.checkpoint_path, map_location=ptu.device)
        net_kwargs = self.variant["net_kwargs"]
        self.test_model = create_segmenter(net_kwargs)
        self.test_model.load_state_dict(checkpoint["model"], strict=True)

        return self.checkpoint_path

    def evaluate(self, data, model_path, **kwargs):
        if data is None or data.x is None or data.y is None:
            raise Exception("Prediction data is None")

        # make test loader
        dataset_dir = Path(data.x[0].split("/image_file_index")[0]) # temporary solution
        dataset_kwargs = self.variant["dataset_kwargs"]
        test_kwargs = dataset_kwargs.copy()
        test_kwargs["split"] = "test"
        test_kwargs["batch_size"] = 2
        test_kwargs["crop"] = False
        test_loader = create_dataset(dataset_dir, test_kwargs)
        n_cls = test_loader.unwrapped.n_cls

        test_seg_gt = test_loader.dataset.get_gt_seg_maps()

        ptu.set_gpu_mode(True)
        distributed.init_process()
        self.load(model_path)
        self.test_model.eval()
        self.test_model.to(ptu.device)

        if ptu.distributed:
            self.test_model = DDP(self.test_model, device_ids=[ptu.device], find_unused_parameters=True)

        test_pred_maps = self.predict(test_loader, save_images=False)

        distributed.barrier()
        distributed.destroy_process()
        metric_name, metric_func = kwargs.get("metric")

        if callable(metric_func):
            return {"miou_score": metric_func(test_seg_gt, test_pred_maps, n_cls)}
        else:
            raise Exception(f"not found model metric func(name={metric_name}) in model eval phase")
