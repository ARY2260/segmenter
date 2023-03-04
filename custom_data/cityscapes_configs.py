algorithm_kwargs = {'batch_size': 1,
  'eval_freq': 8,
  'num_epochs': 216,
  'start_epoch': 214}

dataset_kwargs = {'batch_size': 1,
  'crop_size': 768,
  'dataset': 'cityscapes',
  'image_size': 1024,
  'normalization': 'vit',
  'num_workers': 10,
  'split': 'train'}

inference_kwargs = {'im_size': 1024,
  'window_size': 768,
  'window_stride': 512}

net_kwargs = {'backbone': 'vit_large_patch16_384',
  'd_model': 1024,
  'decoder': {'drop_path_rate': 0.0,
           'dropout': 0.1,
           'n_cls': 19,
           'n_layers': 1,
           'name': 'mask_transformer'},
  'drop_path_rate': 0.1,
  'dropout': 0.0,
  'image_size': (768, 768),
  'n_cls': 19,
  'n_heads': 16,
  'n_layers': 24,
  'normalization': 'vit',
  'patch_size': 16}

optimizer_kwargs = {'clip_grad': None,
  'epochs': 216,
  'iter_max': 80352,
  'iter_warmup': 0.0,
  'lr': 0.01,
  'min_lr': 1e-05,
  'momentum': 0.9,
  'opt': 'sgd',
  'poly_power': 0.9,
  'poly_step_size': 1,
  'sched': 'polynomial',
  'weight_decay': 0.0}