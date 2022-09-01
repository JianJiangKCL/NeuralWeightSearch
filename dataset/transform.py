from torchvision import transforms
import numpy as np
import torch

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

mean_cifar = [0.4914, 0.4822, 0.4465]
std_cifar = [0.2023, 0.1994, 0.2010]
transform_train_cifar = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_cifar, std=std_cifar)
])

transform_test_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_cifar, std=std_cifar)
])

mean_image224 = [0.485, 0.456, 0.406]
std_image224 = [0.229, 0.224, 0.225]
transform_image224_train = transforms.Compose([

    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_image224, std=std_image224)
])

transform_image224_test = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
 transforms.ToTensor(),
transforms.Normalize(mean=mean_image224, std=std_image224)
])

transform_image224_train_aug = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_image224, std=std_image224),
    Cutout(16)
])

transform_image224_test_aug = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
 transforms.ToTensor(),
transforms.Normalize(mean=mean_image224, std=std_image224)
])