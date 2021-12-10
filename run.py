import argparse
import os
import shutil

from mmcv import Config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler

from msp import LightningModel
from msp.utils import setup_seed, partial_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train or test a detector.")
    parser.add_argument("config", help="Train config file path.")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    setup_seed(cfg.random_seed)

    model = LightningModel(cfg)

    checkpoint_callback = ModelCheckpoint(
        filepath=f"{cfg.checkpoint_path}/{cfg.name}/{cfg.version}/"
                 f"{cfg.name}_{cfg.version}_{{epoch}}_{{avg_val_loss:.3f}}_{{ade:.3f}}_{{fde:.3f}}_{{fiou:.3f}}",
        save_last=None,
        save_top_k=-1,
        verbose=True,
        monitor='fiou',
        mode='max',
        prefix=''
    )

    lr_logger_callback = LearningRateLogger(logging_interval='step')
    profiler = SimpleProfiler() if cfg.simple_profiler else AdvancedProfiler()
    logger = TensorBoardLogger(save_dir=cfg.log_path, name=cfg.name, version=cfg.version)

    trainer = pl.Trainer(
        gpus=cfg.num_gpus,
        #distributed_backend='dp',
        max_epochs=cfg.max_epochs,
        logger=logger,
        profiler=profiler,
        callbacks=[lr_logger_callback],
        #gradient_clip_val=cfg.gradient_clip_val,\
        checkpoint_callback=checkpoint_callback,
        check_val_every_n_epoch=10,
        resume_from_checkpoint=cfg.resume_from_checkpoint,
        accumulate_grad_batches=cfg.batch_accumulate_size) # 由于每个batch内只有一个样本，所以采用这样的处理方法

    if (not (args.train or args.test)) or args.train:

        shutil.copy(args.config, os.path.join(cfg.log_path, cfg.name, cfg.version, args.config.split('/')[-1]))

        if cfg.load_from_checkpoint is not None:
            model_ckpt = partial_state_dict(model, cfg.load_from_checkpoint)
            model.load_state_dict(model_ckpt)

        trainer.fit(model)

    if args.test:
        if cfg.test_checkpoint is not None:
            model_ckpt = partial_state_dict(model, cfg.test_checkpoint)
            model.load_state_dict(model_ckpt)

        trainer.test(model)


if __name__ == '__main__':
    main()
