import os
from pathlib import Path
import argparse
import pandas as pd
import torch


def cal_bits(file_path):
    file_size = os.path.getsize(file_path)
    return file_size * 8


def cal_MB(file_path):
    file_size = os.path.getsize(file_path)
    return file_size / 1024 / 1024


def cal4Nws(root, benchmark='cifar'):
    total_code_size = 0
    total_bn_size = 0
    flag_cifar = False
    for path in Path(root).rglob('*.lzma'):

        print(path)
        total_code_size += cal_MB(path)
    for path in Path(root).rglob('*_bns*'):
        if '19' in path.stem:
            flag_cifar = True
        total_bn_size += cal_MB(path)
    total_size = total_code_size + total_bn_size
    print('total_code_size: in MB', total_code_size
          , 'total_bn_size:', total_bn_size
          , 'per task size:', total_size/20 if flag_cifar else total_size/5)

    kp_size = 0
    for path in Path(root).rglob('*shared_kernel_pools*'):
        kp_size += cal_MB(path)
        break

    print('total_size:', total_size + kp_size)


def cal4KSM(root):
    total_size = 0
    size_list = []
    for path in Path(root).rglob('*.pt'):
        print(path)
        current_size = cal_MB(path)
        size_list.append(current_size)
        total_size += current_size
    df = pd.DataFrame({'size': size_list})
    df.to_csv(os.path.join(root, 'size.csv'), index=False)
    print(total_size)


def calPackNet(root):
    try:
        state20 = torch.load(f'{root}/vehicles_2/one_shot_prune/checkpoint-30.pth.tar')
    except:
        state20 = torch.load(f'{root}/processed_food/one_shot_prune/checkpoint-30.pth.tar')
    save_path = root + '/'
    total_masks = state20['masks']
    # maks = total_masks['module.conv1'][0]
    torch.save(total_masks, save_path + 'total_masks.pt')
    total_shared_layer = state20['shared_layer_info']
    torch.save(total_shared_layer, save_path + 'total_shared_layer.pt')
    backbone = state20['model_state_dict']
    torch.save(backbone, save_path + 'backbone.pt')
    backbone_size = cal_MB(Path(save_path + 'backbone.pt'))
    print('backbone_size ', backbone_size)
    total_masks_size = cal_MB(Path(save_path + 'total_masks.pt'))
    print('total_masks_size ', total_masks_size)
    total_shared_layer_size = cal_MB(Path(save_path + 'total_shared_layer.pt'))
    print('total_shared_layer_size ', total_shared_layer_size)


def calCPG(root):
    try:
        state20 = torch.load(f'{root}/scratch_mul_1.5/resnet18/vehicles_2/gradual_prune/checkpoint-100.pth.tar')
    except:
        # state20 = torch.load(f'{root}/0.5/checkpoint-4.pth.tar')
        state20 = torch.load(f'{root}/checkpoint-100.pth.tar')
    save_path = root + '/'
    total_masks = state20['masks']
    # maks = total_masks['module.conv1'][0]
    torch.save(total_masks, save_path + 'total_masks.pt')
    total_shared_layer = state20['shared_layer_info']
    torch.save(total_shared_layer, save_path + 'total_shared_layer.pt')
    backbone = state20['model_state_dict']
    torch.save(backbone, save_path + 'backbone.pt')
    backbone_size = cal_MB(Path(save_path + 'backbone.pt'))
    print('backbone_size ', backbone_size)
    total_masks_size = cal_MB(Path(save_path + 'total_masks.pt'))
    print('total_masks_size ', total_masks_size)
    total_shared_layer_size = cal_MB(Path(save_path + 'total_shared_layer.pt'))
    print('total_shared_layer_size ', total_shared_layer_size)


def cal4AQD(root):
    total_size = 0
    size_list = []
    cnt = 0
    for path in Path(root).rglob('*.lzma'):
        print(path)
        current_size = cal_MB(path)
        size_list.append(current_size)
        total_size += current_size
        cnt += 1
    for path in Path(root).rglob('*float_weights'):
        print(path)
        current_size = cal_MB(path)
        size_list.append(current_size)
        total_size += current_size
        cnt += 1
    df = pd.DataFrame({'ACC': size_list})
    df.to_csv(os.path.join(root, 'acc.csv'), index=False)
    print('number of files ', cnt)
    print('total_size ', total_size)


def main(args):

    root = args.root
    if args.setting == 'NWS':
        cal4Nws(root, args.benchmark)
    elif args.setting == 'KSM':
        cal4KSM(root)
    elif args.setting == 'AQD':
        cal4AQD(root)
    elif args.setting == 'PACKNET':
        calPackNet(root)
    elif args.setting == 'CPG':
        calCPG(root)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--setting", default='NWS', type=str)
    parser.add_argument("--benchmark", default='cifar', type=str)
    args = parser.parse_args()
    main(args)

