import os.path
import wandb

from dataset.cifar100_dataset import cifar100_train_loader, cifar100_val_loader
from dataset.fine_grained_dataset import ds_train_loader, ds_val_loader
from funcs.utils_funcs import *
from funcs.test_reconstructed_model import test_codes
from models.NwsTrainer import NwsTrainer
from funcs.setup import parse_args, set_logger, set_trainer
from models.quantizer import Quantize
import torch.nn as nn


def main(args):
    if 'resnet' in args.arch:
        import models.NWS as NWS
    else:
        import models.NWS_v2 as NWS
    if args.dataset == 'cifar':
        exp_indicator = 'cifar'
    else:
        # use the same root folder for fine-grained dataset
        exp_indicator = 'cub2sketches'
    root_dir = f"{args.results_dir}/{exp_indicator}_{args.arch}_lr{args.lr}_e{args.epoch}_nemb{args.n_emb}_beta{args.beta}_seed{args.seed}"
    save_path = os.path.join(root_dir, f'task{args.task_id}')
    os.makedirs(save_path, exist_ok=True)
    device = "cuda"

    model = NwsTrainer(args)
    backbone = model.backbone

    if args.finetune:
        if args.finetune == 'previous_task':
            args.finetune = os.path.join(root_dir, f'task{args.task_id-1}')
        if args.use_recon_codes:
            print('--------------use previous recon weights')
            cnn_out_in = NWS.get_out_in_size(backbone)
            codes_path = os.path.join(root_dir, f'{args.task_id - 1}task_codes.lzma')
            codes_layers = NWS.load_codes_from_lzma(codes_path, cnn_out_in)
            qtzs_in_order = torch.load(f"{root_dir}/shared_kernel_pools.pt")
            bns = torch.load(f"{root_dir}/task{args.task_id - 1}_bns.pt")
            NWS.load_codes2weights_individual_(backbone, codes_layers, cnn_out_in, qtzs_in_order, bns)
            backbone.load_qtz(qtzs_in_order)
            backbone.reset_last_layer(args.end_class, finetune=True)
        else:
            print('--------------use previous temporal weights')
            checkpoint = torch.load(args.finetune)
            backbone = load_state_from_ddp(backbone, checkpoint['model'])
            backbone.reset_last_layer(args.end_class, finetune=True)
            if args.use_qtz_only:
                print('--------------empty previous temporal weights; use qtz only')
                backbone.initial_conv()
                qtzs_in_order = []
                for module in backbone.modules():
                    if isinstance(module, Quantize):
                        qtzs_in_order.append(module)
                backbone.load_qtz(qtzs_in_order)

    if args.dataset == 'cifar':
        train_loader = cifar100_train_loader(args.dataset_path, args.task_id, args.batch_size, args.num_workers)
        test_loader = cifar100_val_loader(args.dataset_path, args.task_id, args.batch_size, args.num_workers)
    else:
        train_loader = ds_train_loader(os.path.join(args.dataset_path, 'train'), args.batch_size, args.num_workers)
        test_loader = ds_val_loader(os.path.join(args.dataset_path, 'test'), args.batch_size, args.num_workers)

    wandb_logger = set_logger(args, root_dir)
    trainer = set_trainer(args, wandb_logger, save_path)

    trainer.fit(model, train_loader)
    print('--------------finish training')

    trainer.save_checkpoint(f'{save_path}/checkpoint.pt')

    cnn_out_in = NWS.get_out_in_size(backbone)
    flattened_model_codes, unique_codes_layers = backbone.encode_weights()

    # save model kernel indices
    file_name = f'{args.task_id}task_codes'
    file_name = os.path.join(root_dir, file_name)
    NWS.save_model_codes(flattened_model_codes, file_name)

    qtzs_in_order = []
    bns = []
    for module in backbone.modules():
        if isinstance(module, Quantize):
            qtzs_in_order.append(module.cpu())
        if isinstance(module, nn.BatchNorm2d):
            bns.append(module.cpu())

    if args.task_id == 0:
        torch.save(qtzs_in_order, os.path.join(root_dir, f'shared_kernel_pools.pt'))
    torch.save(bns, os.path.join(root_dir, f'task{args.task_id}_bns.pt'))

    ############################
    # using kernel indices to reconstruct model weights
    # and test the reconstructed model to report accuracy in the paper
    print('----using kernel indices to reconstruct model weights')
    model_codes_layers = NWS.make_codes_unflatten(flattened_model_codes, cnn_out_in)
    NWS.load_codes2weights_individual_(backbone, model_codes_layers, cnn_out_in, qtzs_in_order, bns)
    backbone = backbone.cuda()

    recon_acc = test_codes(test_loader, backbone, device)
    wandb.summary['recon_test_accuracy'] = recon_acc
    file_name = f'{root_dir}/recon_acc.txt'
    with open(file_name, 'a') as f:
        f.write(f'{recon_acc}\n')

    trainer.test(model, test_loader)
    wandb.finish()
    ##########


if __name__ == "__main__":
    args = parse_args()
    # set random seed
    set_seed(args.seed)
    print(args)
    main(args)
