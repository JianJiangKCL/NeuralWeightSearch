import argparse
from args.general.dataset import dataset_args
from args.general.train import train_args
from args.general.project import project_args
from args.custom import custom_args
from funcs.utils_funcs import load_temp_, load_yaml_
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import StochasticWeightAveraging, LearningRateMonitor


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    # add shared arguments
    dataset_args(parser)
    project_args(parser)
    train_args(parser)

    # add custom args
    custom_args(parser)

    # parse args
    args = parser.parse_args()

    # load config
    if args.config_file is not None:

        load_yaml_(args)
    # update args if  temp_args is given
    load_temp_(args)
    #
    return args


def set_logger(args, root_dir):
    wandb_logger = WandbLogger(
        name=f"{args.name}-task{args.task_id}",
        project=args.project,
        offline=True if args.wandb_mode == 'offline' else False,
        reinit=True,
        save_dir=root_dir,
        config=args,
    )
    wandb_logger.log_hyperparams(args)
    return wandb_logger


def set_trainer(args, logger, save_path, kwargs=None):
    extra_kwargs = kwargs
    callbacks = []
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, save_last=True, monitor='val_loss',
                                          mode='min', save_top_k=3)
    callbacks.append(checkpoint_callback)

    if args.lr_logger:
        lr_callback = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_callback)
    if args.use_swa:
        swa_callback = StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=0.05)
        callbacks.append(swa_callback)

    # grad_clip cannot solve nan in this case
    if args.grad_clip:
        print('Using gradient clipping')
        gradient_clip_val = 0.5
    else:
        gradient_clip_val = 0.0

    kwargs = {'logger': logger, 'enable_checkpointing': True, 'max_epochs': args.epochs,
              'check_val_every_n_epoch': 10, 'callbacks': callbacks, 'gradient_clip_val': gradient_clip_val, 'gradient_clip_algorithm': "value", 'detect_anomaly': False}

    if extra_kwargs is not None:
        kwargs.update(extra_kwargs)

    if args.disable_tqdm:
        kwargs['enable_progress_bar'] = False

    if args.distributed:
        kwargs['gpus'] = args.gpus
        kwargs['strategy'] = 'ddp'
        kwargs['sync_batchnorm'] = True
    else:
        kwargs['gpus'] = 1

    if args.use_amp:
        kwargs['precision'] = 16
        # in default amp_backend is 'native'
        kwargs['amp_backend'] = 'native'
    else:
        kwargs['precision'] = 32

    print('kwargs for the trainer: ', kwargs)
    trainer = Trainer.from_argparse_args(args, **kwargs)
    return trainer
