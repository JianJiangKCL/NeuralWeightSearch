import argparse
import json
import lzma
import os.path
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torch.multiprocessing as mp
from torch import optim
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, ImageNet, ImageFolder

from models.NWS_v2 import NwsConv
from models.NwsResNet import Nws_resnet18, Nws_resnet34
from models.quantizer import Quantize
from funcs.utils_funcs import *
from dataset.transform import *
from models.NwsMobileNet import NwsMobileNetV2
from models.NwsVGG import Nwsvgg16_bn
from tqdm import tqdm

'''
Pure torch version DDP for pretraining kernel pools.
If nan loss is detected, please consider avoiding amp16 that may cause overflow.
'''


@torch.no_grad()
def test_AQ(epochs, loader, model, device, root_process, args):
    loader = tqdm(loader, disable=(not root_process) or args.disable_tqdm)

    acc_sum = 0
    n_sum = 0
    # freeze the quantizer everytime, otherwise model.train() will override their training mode
    beta = args.beta
    model.eval()

    for x, y in loader:
        w_diffs = []
        # y = y.squeeze(-1).to(device)
        y = y.to(device)
        x = x.to(device)

        # zero grad
        model.zero_grad()

        logits = model(x)
        logits = logits.view(logits.size(0), -1)

        for module in model.modules():
            # this is used to update the model weights
            if isinstance(module, NwsConv):
                w_diffs.append(module.diff)
        w_diff = sum(w_diffs)
        _, winners = (logits).max(1)
        acc = torch.sum((winners == y).int())

        n_sum += y.size(0)
        if args.distributed:

            reduced_n_sum = torch.tensor(n_sum).to(device)

            dist.reduce(acc, dist.ReduceOp.SUM)
            dist.reduce(reduced_n_sum, dist.ReduceOp.SUM)
            reduced_n_sum = reduced_n_sum.item()

        acc_sum += acc.detach().item()
        if args.distributed:
            avg_acc = acc_sum / reduced_n_sum
        else:
            avg_acc = acc_sum / n_sum

        loader.set_description(
            (
                f"epochs: {epochs + 1}; "
                f" acc:{avg_acc:.5f} ;"
                f" w_diff :{w_diff:.5f} "
            )
        )
    if root_process:
        logging.info('====> test epochs{}:train_acc {}'.format(epochs, avg_acc))
    return avg_acc


def train_AQ(epochs, loader, model, opt, scaler, device, root_process, args):
    loader = tqdm(loader, disable=(not root_process) or args.disable_tqdm)
    criterion = nn.CrossEntropyLoss()
    acc_sum = 0
    n_sum = 0
    # freeze the quantizer everytime, otherwise model.train() will override their training mode
    beta = args.beta
    model.train()

    for x, y in loader:
        w_diffs = []
        y = y.squeeze(-1).to(device)
        x = x.to(device)

        # zero grad
        model.zero_grad()

        with autocast():

            logits = model(x)
            # squeeze
            # logits = rearrange(logits, 'b n_c () () -> b n_c')
            logits = torch.squeeze(torch.squeeze(logits, dim=-1), dim=-1)
            # this code is the same to model.moduel.modules()
            for module in model.modules():
                # this is used to update the model weights
                if isinstance(module, NwsConv):
                    # print(module)
                    w_diffs.append(module.diff)
            clf_loss = criterion(logits, y)
            w_diff = sum(w_diffs)
            diff = w_diff
            loss = clf_loss + diff * beta

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        _, winners = (logits).max(1)
        acc = torch.sum((winners == y).int())

        acc_sum += acc.detach().item()
        n_sum += y.size(0)

        if isinstance(model, DDP):
            model.module.update_qtz()
        else:
            model.update_qtz()
        avg_acc = acc_sum / n_sum

        lr = opt.param_groups[0]['lr']
        loader.set_description(
            (
                f"epochs: {epochs + 1}; "
                f" acc:{avg_acc:.5f} ;"
                f" lr:{lr:.5f} ;"
                f" ce :{clf_loss.item():.5f} "
                f" w_diff :{w_diff:.5f} "
            )
        )
    logging.info('====> train epochs{}:train_acc {}'.format(epochs, avg_acc))
    return avg_acc


def get_model_codes(model):
    codes = model.module.encode_weights()
    return codes


def main(local_rank, args):
    root = args.root
    results_dir = f"{args.results_dir}/{args.dataset}_{args.arch}/lr{args.lr}" \
                  f"_e{args.epochs}_nemb{args.n_emb}_epoch{args.epochs}_gs{args.gs}"

    save_path = os.path.join(root, results_dir)
    args.save_path = save_path
    args.local_rank = local_rank
    device = "cuda"
    # print cuda info
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    distributed = args.distributed
    if distributed:
        torch.backends.cudnn.enabled = True
        args.world_size = args.gpus
        print('Initializing process Process Group {}'.format(args.local_rank))
        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size,
                             rank=args.local_rank)
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        args.batch_size = int(args.batch_size / args.world_size)
        args.num_workers = int((args.num_workers + args.world_size - 1) / args.world_size)
        print('device {}, batchsize {}, numworkers {}'.format(device, args.batch_size, args.num_workers))

    root_process = True

    if distributed and not torch.distributed.get_rank() == 0:
        root_process = False
    if root_process:
        os.makedirs(save_path, exist_ok=True)
        json_file_name = os.path.join(save_path, 'args.json')
        with open(json_file_name, 'w') as fp:
            json.dump(dict(args._get_kwargs()), fp, sort_keys=True, indent=4)
        checkpoints_path = os.path.join(save_path, 'checkpoints')
        os.makedirs(checkpoints_path, exist_ok=True)
        sample_output_path = os.path.join(save_path, 'output')
        os.makedirs(sample_output_path, exist_ok=True)

        log_file = os.path.join(save_path, 'log.txt')

        config_logging(log_file, args.resume)
        logging.info('====>  args{} '.format(args))

    if args.dataset == 'cifar':
        end_class = 100
    else:
        end_class = 1000

    if root_process:
        print('root is', root)
        print('save_path is:', save_path)
        # print("Load model")
        # print('data cls', classes)
        print('args-------', args)

    model = None
    # to gpu when necessary
    if args.arch == 'resnet18':
        model = Nws_resnet18(args.n_emb, num_classes=end_class, gs=args.gs).cuda()
    elif args.arch == 'resnet34':
        model = Nws_resnet34(args.n_emb, num_classes=end_class, gs=args.gs).cuda()
    elif args.arch == 'mobilenetv2':
        model = NwsMobileNetV2(args.n_emb, num_classes=end_class).cuda()
    elif args.arch == 'vgg16':
        model = Nwsvgg16_bn(args.n_emb, num_classes=end_class).cuda()
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    epochs = args.epochs

    op_multi = lambda a, b: int(a * b)
    if args.optimizer == 'adam':
        opt = optim.Adam(model.parameters(), lr=args.lr)
        MILESTONES = list((map(op_multi, [0.5], [epochs])))
    elif args.optimizer == 'sgd':
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-5, nesterov=True)
        MILESTONES = list((map(op_multi, [0.5, 0.8], [epochs, epochs])))
        if args.epochs == 160:
            MILESTONES = [100]
    scheduler = MultiStepLR(opt, milestones=MILESTONES, gamma=0.1)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device if distributed else None)
        args.start_epoch = checkpoint['epochs']
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if root_process:
            logging.info("=> loading checkpoint '{}', epochs{}".format(args.resume, args.start_epoch))

    if args.dataset == 'cifar':
        train_ds = CIFAR100(root=args.dataset_path, train=True, download=False, transform=transform_train_cifar)
        test_ds = CIFAR100(root=args.dataset_path, train=False, download=False, transform=transform_test_cifar)
    elif args.dataset == 'imagenet':
        train_ds = ImageNet(root=args.dataset_path, split='train', download=False, transform=transform_image224_train)
        test_ds = ImageNet(root=args.dataset_path, split='val', download=False, transform=transform_image224_test)
    elif args.dataset == 'subimagenet':
        train_path = os.path.join(args.dataset_path, 'train')
        train_ds = ImageFolder(train_path, transform=transform_image224_train)

        test_dataset_path = os.path.join(args.dataset_path, 'val')
        test_ds = ImageFolder(test_dataset_path, transform=transform_image224_test)

    # if distributed over multiple GPU's, set-up barrier a barrier ensuring that all the processes have loaded the data
    if distributed:
        dist.barrier()

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_ds, num_replicas=dist.get_world_size(), rank=dist.get_rank())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                              drop_last=False, pin_memory=True, sampler=train_sampler if
        distributed else None, shuffle=False if distributed else True)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                             drop_last=False, pin_memory=True, sampler=test_sampler if
        distributed else None, shuffle=False)

    # amp16 may induce overflow, causing nan loss.
    scaler = GradScaler()
    for epochs in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epochs)
        train_AQ(epochs, train_loader, model, opt, scaler, device, root_process, args)
        if (epochs + 1) % 10 == 0 or epochs == args.epochs - 1:
            test_AQ(epochs, test_loader, model, device, root_process, args)
        scheduler.step()

        if root_process:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epochs': epochs + 1
            }
            # save the last current
            pt_path = os.path.join(save_path, f"checkpoints/KP_resume.pt")
            torch.save(checkpoint, pt_path)
            if (epochs + 1) % 5 == 0:
                pt_path = os.path.join(save_path, f"checkpoints/KP_{epochs + 1}.pt")
                torch.save(checkpoint, pt_path)

    if root_process:
        model_codes, unique_codes_layers = get_model_codes(model)

        out_in = []
        for module in model.modules():
            if isinstance(module, NwsConv):
                out_in.append(module.weight.size())

        torch.save(out_in, os.path.join(save_path, 'out_in'))
        model_codes = model_codes.numpy().astype(np.int16)

        np.savez(os.path.join(save_path, 'model_codes'), model_codes=model_codes)
        ### validate the equality of lzma compression
        # model_codes = rearrange(model_codes, 't len -> (t len)')
        with lzma.open(os.path.join(save_path, 'model_codes_lzma'), 'wb') as f:
            f.write(lzma.compress(model_codes))

        # saving their state_dict may also be an option, but it's hard to combining them as a list.
        qtzs_in_order = []
        bns = []
        for module in model.modules():
            if isinstance(module, Quantize):
                qtzs_in_order.append(module.cpu())
            if isinstance(module, nn.BatchNorm2d):
                bns.append(module.cpu())
        torch.save(qtzs_in_order, os.path.join(save_path, 'shared_kernel_pools.pt'))

        torch.save(bns, os.path.join(save_path, 'bns.ckpt'))


########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--arch", type=str, default='resnet18', help="architecture of the model")
    parser.add_argument("--n_emb", type=int, help=' the size of codebook, i.e. the number of embeddings', default=512)
    parser.add_argument("--beta", default=0.5, type=float)
    parser.add_argument("--gs", default=1, type=int, help="group size")

    # train
    parser.add_argument('-c', '--config-file', required=True, type=str, help="xxx.yaml")
    parser.add_argument('-t', '--temporary', action='append', type=str, help="dynamic change args")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--optimizer", default='adam', type=str)

    parser.add_argument('--start_epoch', default=0, type=int, help='manual epochs number (useful on restarts)')
    parser.add_argument("--disable_tqdm", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--distributed", type=int,
                        help="distribute (1) over different gpu's or not (0)", default=0)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--dist_url', default='tcp://localhost:12335', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("--batch_size", default=128, type=int, help="number of data samples in the mini_batch")

    # data
    parser.add_argument("--dataset", default='cifar', type=str)
    parser.add_argument("--dataset_path", default='D:\Dataset\cifar100', type=str)
    parser.add_argument("--root", default='', type=str)
    parser.add_argument("--results_dir", default='results', type=str)

    args = parser.parse_args()

    load_yaml_(args)
    load_temp_(args)

    set_seed(args.seed)
    # print(args)
    if args.distributed:
        mp.spawn(main,
                 nprocs=args.gpus,
                 args=(args,),
                 join=True)
    else:
        main(0, args)
