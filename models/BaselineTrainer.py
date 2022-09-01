import pytorch_lightning as pl
import torch.nn as nn

from pytorch_lightning.utilities import AttributeDict
import torchmetrics

from torchvision.models import resnet18, mobilenet_v2, vgg16_bn
from funcs.module_funcs import setup_optimizer, setup_scheduler


class BaselineTrainer(pl.LightningModule):
    args: AttributeDict
    def __init__(self, args):
        super(BaselineTrainer, self).__init__()
        self.acc_sum = 0
        self.n_sum = 0
        self.train_accuracy = torchmetrics.Accuracy(top_k=1)
        self.test_accuracy = torchmetrics.Accuracy(top_k=1)
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        arch = args.arch
        self.root_dir = args.root_dir
        num_classes = args.end_class
        if arch == 'resnet18':
            self.backbone = resnet18(pretrained=True)
            self.backbone.fc = nn.Linear(512, num_classes)
        elif arch == 'mobilenetv2':
            self.backbone = mobilenet_v2(pretrained=True)
            self.backbone.classifier = nn.Sequential(
                # nn.Dropout(p=0.2), # omit dropout for mobilenetv2 to improve its performance
                nn.Linear(self.backbone.last_channel, num_classes),
            )
        elif arch == 'vgg16':
            self.backbone = vgg16_bn(pretrained=True)
            self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.backbone.classifier = nn.Sequential(
                nn.Linear(512, num_classes),
            )
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
        loss = ce_loss
        self.train_accuracy.update(y_hat, y)
        acc = self.train_accuracy.compute()
        log_data = {
            'loss': loss,
            'ce_loss': ce_loss,
            'acc': acc,
        }

        self.log_dict(log_data, prog_bar=not self.args.disable_tqdm)
        return loss

    def training_epoch_end(self, outputs):
        print('accuracy:', self.train_accuracy.compute())
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        test_loss = self.criterion(y_hat, y)
        self.test_accuracy.update(y_hat, y)
        acc = self.test_accuracy.compute()
        log_data = {
            'test_loss': test_loss,
            'test_acc': acc
        }

        self.log_dict(log_data, prog_bar=True, on_epoch=True, on_step=False)

    def test_epoch_end(self, outputs):

        test_acc = self.test_accuracy.compute()
        file_name = f'{self.root_dir}/recon_acc.txt'
        with open(file_name, 'a') as f:
            f.write(f'{self.args.task_id} {test_acc}\n')
        self.test_accuracy.reset()

