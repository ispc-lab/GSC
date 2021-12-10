import time
import numpy as np
import pytorch_lightning as pl
import torch.optim as optim
import torch
from mmcv.utils import Config
from torch.utils.data import DataLoader
from .models import build_predictor
from .datasets.utils.metrics import evaluate_metric
from .datasets import build_dataset
from matplotlib import pyplot as plt

import os
from .utils.train_utils import summarize_metric

from torch.utils.tensorboard import SummaryWriter

class LightningModel(pl.LightningModule):

    def __init__(self, cfg: Config):
        super(LightningModel, self).__init__()

        self.cfg = cfg
        self.model_cfg = cfg.model
        self.data_cfg = cfg.data
        self.optim_cfg = cfg.optimizer_cfg
        self.lr_cfg = cfg.lr_cfg

        self.model = build_predictor(self.model_cfg)
        self.hparams = dict(lr=self.optim_cfg.lr,
                            batch_size=self.data_cfg.batch_size * cfg.batch_accumulate_size)

        self.whether_save = True
        if self.whether_save:
            self.writer = SummaryWriter(comment="Training")
            self.writer.add_hparams({'lr':self.optim_cfg.lr, 'bsize':self.data_cfg.batch_size * cfg.batch_accumulate_size}, {})
        self.train_step = 1
        self.val_step = 1
        self.tests_step = 1

    def forward(self, data, **kwargs):
        pred, masker_pred = self.model(data, **kwargs)
        return pred, masker_pred

    def training_step(self, batch, batch_idx):
        data = batch
        pred, masker_pred = self(data, module="Train")
        acdt_loss, masker_loss = self.model.loss(pred, masker_pred, data)
        loss = acdt_loss + masker_loss        
        
        with torch.no_grad():
            masker_pred_np = masker_pred.cpu().numpy()
        
        shelter_precision = (masker_pred_np[masker_pred_np[:,-1] == 1., 1] >= 0.5).sum() / (masker_pred_np[:,-1] == 1.).sum()
        go_away_precision = (masker_pred_np[masker_pred_np[:,-1] == 0., 0] >= 0.5).sum() / (masker_pred_np[:,-1] == 0.).sum()
        precision = (np.sum((np.where(masker_pred_np[:, :2] >= 0.50)[1] == masker_pred_np[:, -1])!=0)) / masker_pred_np.shape[0]

        if self.whether_save:
            self.writer.add_scalar('Training/acdt_loss', acdt_loss, self.train_step)
            self.writer.add_scalar('Training/masker_loss', masker_loss, self.train_step)
            self.writer.add_scalar('Training/shelter_precision', shelter_precision, self.train_step)
            self.writer.add_scalar('Training/go_away_precision', go_away_precision, self.train_step)
            self.writer.add_scalar('Training/precision', precision, self.train_step)


        logs = {"loss": loss}
        self.train_step += 1

        return {"loss": loss,
                "log": logs}

    def validation_step(self, batch, batch_idx):
        data = batch
        pred_np, pred_gate_np = self.model.predict(data, module="Test")
        ap, mtta, tta_r80, p_r_plot = evaluate_metric(pred_np, data)

        self.val_step += 1
        
        ap_list = []
        mtta_list = []
        tta_r80_list = []

        ap_list.append(ap)
        mtta_list.append(mtta)
        tta_r80_list.append(tta_r80)

        logs = {"ap":ap_list, "mtta":mtta_list, "tta_r80":tta_r80_list}
        return logs

    def validation_epoch_end(self, output):
        average_ap, average_mtta, average_tta_r80 = summarize_metric(output)
        if self.whether_save:
            self.writer.add_scalar('Metrics/AP', average_ap, self.tests_step)
            self.writer.add_scalar('Metrics/mtta', average_mtta, self.tests_step)
            self.writer.add_scalar('Metrics/TTA_R80', average_tta_r80, self.tests_step)
        self.tests_step += 1
        return {"output": output}

    def test_step(self, batch, batch_idx):
        data = batch
        pred_np = self.model.predict(data, module="Test")
        for i in range(data['location'].shape[0]):
            root_fill = './results/' + str(149) + "/"
            if not os.path.exists(root_fill):
                os.makedirs(root_fill)
            file_name = str(data["video_id"][i]) + ".csv"
            np.savetxt(root_fill + file_name, pred_np[i][:,1], delimiter=',')

        return None

    def prepare_data(self):
        self.train_dataset = build_dataset(self.data_cfg.train)
        self.val_dataset = build_dataset(self.data_cfg.val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.data_cfg.batch_size,
                          shuffle=False,
                          num_workers=self.data_cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.data_cfg.batch_size,
                          shuffle=False,
                          num_workers=self.data_cfg.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.data_cfg.num_workers)

    def configure_optimizers(self):
        
        optim_cfg = self.optim_cfg.copy()
        optim_class = getattr(optim, optim_cfg.pop("type"))
        optimizer = optim_class(self.parameters(), **optim_cfg)

        lr_cfg = self.lr_cfg.copy()
        lr_sheduler_class = getattr(optim.lr_scheduler, lr_cfg.pop("type"))
        scheduler = {
            "scheduler": lr_sheduler_class(optimizer, **lr_cfg),
            "monitor": 'avg_val_loss',
            "interval": "epoch",
            "frequency": 1
        }

        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure = None,
                       on_tpu: bool = False, using_native_amp=False, using_lbfgs: bool = False):

        warm_up_type, warm_up_step = self.cfg.warm_up_cfg.type, self.cfg.warm_up_cfg.step_size
        if warm_up_type == 'Exponential':
            lr_scale = self.model_cfg.spatial_graph.hidden_feature ** -0.5
            lr_scale *= min((self.trainer.global_step + 1) ** (-0.5),
                            (self.trainer.global_step + 1) * warm_up_step ** (-1.5))
        elif warm_up_type == "Linear":
            lr_scale = min(1., float(self.trainer.global_step + 1) / warm_up_step)
        else:
            raise NotImplementedError
        
        for pg in optimizer.param_groups:
            # import pdb; pdb.set_trace()
            if self.whether_save:
                self.writer.add_scalar('Training/lr', pg['lr'], self.train_step)
            pg['lr'] = lr_scale * self.hparams.lr

        optimizer.step()
        optimizer.zero_grad()






