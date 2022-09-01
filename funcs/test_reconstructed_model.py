import argparse
import os.path

from models.NwsResNet import Nws_resnet18
import torch.multiprocessing as mp
import models.NWS as NWS
import models.NWS_v2 as Nws2
from dataset.cifar100_dataset import cifar100_val_loader
from dataset.fine_grained_dataset import ds_val_loader
from funcs.utils_funcs import *
from models.NwsMobileNet import *
from tqdm import tqdm
# cudnn.deterministic = True


@torch.no_grad()
def test_codes(loader, model, device):
    loader = tqdm(loader)
    acc_sum = 0
    n_sum = 0
    model.eval()
    # eval_mode(True) is necessary when reloading and test if model is constructed from codes;
    # eval_mode(False) is necessary when reloading and test if model load from ckpt and temporary weights;
    model.eval_mode(True)
    for x, y in loader:
        y = y.to(device)
        x = x.to(device)

        logits = model(x)
        logits = logits.view(logits.size(0), -1)

        _, winners = (logits).max(1)

        acc = torch.sum((winners == y).int())

        acc_sum += acc.detach().item()
        n_sum += y.size(0)
        avg_acc = acc_sum / n_sum

        loader.set_description(
            (
                f" acc:{avg_acc:.5f} ;"
            )
        )
    logging.info('====> Evaluation using reconstructed model: acc {}'.format(avg_acc))
    return avg_acc


def main(local_rank, args):

    device = "cuda"

    arch = args.arch
    n_emb = args.n_emb
    num_classes = args.end_class

    if args.arch == 'resnet18':
        get_out_in_size = NWS.get_out_in_size
        load_codes2weights_individual_ = NWS.load_codes2weights_individual_
        model = Nws_resnet18(n_emb, num_classes=num_classes, gs=args.gs)
    elif args.arch == 'mobilenetv2':
        get_out_in_size = Nws2.get_out_in_size
        load_codes2weights_individual_ = Nws2.load_codes2weights_individual_
        model = NwsMobileNetV2(n_emb, num_classes=num_classes)
    # reset num_emb to load the KPs of the previous model

    # way 1; load from temporary weights
    # ckpt = torch.load(args.ckpt_root)
    #
    # model = load_state_from_ddp(model, torch.load(args.ckpt_root)['state_dict'])

    # way 2; load from codes
    qtzs_in_order = torch.load(f"{args.ckpt_root}/task{args.task_id}_qtzs.pt")

    bns = torch.load(f"{args.ckpt_root}/task{args.task_id}_bns.pt")
    codes_path = f'{args.ckpt_root}/{args.task_id}task_codes.lzma'
    cnn_out_in = get_out_in_size(model)
    codes_layers = load_codes_from_lzma(codes_path, cnn_out_in)
    load_codes2weights_individual_(model, codes_layers, cnn_out_in, qtzs_in_order, bns)

    model = model.cuda()

    if args.dataset == 'cifar':
        test_loader = cifar100_val_loader(args.dataset_path, args.task_id, args.batch_size, args.num_workers)

    else:
        test_loader = ds_val_loader(os.path.join(args.dataset_path, 'test'), args.batch_size, args.num_workers)

    # eval_mode(True) is necessary when reloading and test if model is constructed from codes;
    # eval_mode(False) is necessary when reloading and test if model load from ckpt and temporary weights;
    model.eval_mode(True)
    # test_codes(train_loader, model, device, args)
    test_codes(test_loader, model, device)

##########


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default='VGG16')
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--dataset", default='cifar', type=str)
    parser.add_argument("--dataset_path", default='H:\Dataset\cifar100_org', type=str)
    parser.add_argument("--root", default='', type=str)
    parser.add_argument("--ckpt_root", default='', type=str)
    parser.add_argument("--results_dir", default='results', type=str)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--warm", help='warm up epochs', type=int, default=5)
    parser.add_argument("--end_class", help='number of classes used to train classifier', default=5, type=int)
    parser.add_argument("--pretrained_end_class", default=1000, type=int)
    parser.add_argument("--n_emb", type=int, help=' the size of codebook, i.e. the number of embeddings', default=512)
    parser.add_argument("--dim_emb", type=int, help='the dimension of the embedding ', default=64)
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epochs number (useful on restarts)')

    parser.add_argument("--disable_tqdm", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--distributed", type=int,  help="distribute (1) over different gpu's and use Horovod to do so, or not (0)", default=0)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--dist_url', default='tcp://localhost:12335', type=str, help='url used to set up distributed training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--finetune', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("--beta", default=0.5, type=float)
    parser.add_argument("--batch_size", default=128, type=int, help="number of data samples in the mini_batch")
    parser.add_argument("--gs", default=1, type=int, help="group size")
    parser.add_argument("--task_id", default=0, type=int)
    # parser.add_argument("--ckpt_path", default='', type=str)
    parser.add_argument("--use_amp", default=0, type=int)
    parser.add_argument("--use_qtz_only", default=0, type=int)
    parser.add_argument("--freeze_qtz", default=0, type=int)
    parser.add_argument("--expand_ratio", default=0.5, type=float)
    args = parser.parse_args()
    set_seed(args.seed)
    print(args)
    if args.distributed:
        mp.spawn(main,
                 nprocs=args.gpus,
                 args=(args,),
                 join=True)
    else:
        main(0, args)



