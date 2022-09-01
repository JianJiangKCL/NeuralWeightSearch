import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import AttributeDict
from .NwsResNet import Nws_resnet18, Nws_resnet34
from .NwsMobileNet import NwsMobileNetV2
from .NwsVGG import Nwsvgg16_bn
import torchmetrics
from funcs.module_funcs import setup_optimizer, setup_scheduler


class NwsTrainer(pl.LightningModule):
    args: AttributeDict
    
    def __init__(self, args):
        super(NwsTrainer, self).__init__()
        self.acc_sum = 0
        self.n_sum = 0
        self.train_accuracy = torchmetrics.Accuracy(top_k=1)
        self.test_accuracy = torchmetrics.Accuracy(top_k=1)
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        arch = args.arch
        n_emb = args.n_emb
        num_classes = args.pretrained_end_class
        if arch == 'resnet18':
            self.backbone = Nws_resnet18(n_emb, num_classes=num_classes, gs=args.gs)
        elif arch == 'resnet34':
            self.backbone = Nws_resnet34(n_emb, num_classes=num_classes, gs=args.gs)
        elif arch == 'mobilenetv2':
            self.backbone = NwsMobileNetV2(n_emb, num_classes=num_classes)
        elif arch == 'vgg16':
            self.backbone = Nwsvgg16_bn(n_emb,  num_classes=num_classes)
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        opt = setup_optimizer(self.args, self.backbone)
        scheduler = setup_scheduler(self.args, opt)
        return [opt], [scheduler]

    def training_step(self, batch, batch_idx):
 
        x, y = batch
        y_hat = self.backbone(x)

        ce_loss = self.criterion(y_hat, y)

        diff = self.backbone.get_summed_diff()
        loss = ce_loss + diff * self.args.beta

        self.train_accuracy.update(y_hat, y)
        acc = self.train_accuracy.compute()
        log_data = {
            'loss': loss,
            'ce_loss': ce_loss,
            'diff': diff,
            'acc': acc,
        }

        self.log_dict(log_data, rank_zero_only=True, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        acc = self.train_accuracy.compute()
        if self.local_rank == 0:
            print('accuracy:', acc)
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        test_loss = F.cross_entropy(y_hat, y)
        self.test_accuracy.update(y_hat, y)
        acc = self.test_accuracy.compute()
        log_data = {
            'test_loss': test_loss,
            'test_acc': acc
        }

        self.log_dict(log_data, prog_bar=True, on_epoch=True, on_step=False)

    def test_epoch_end(self, outputs):
        test_acc = self.test_accuracy.compute()
        self.test_accuracy.reset()

