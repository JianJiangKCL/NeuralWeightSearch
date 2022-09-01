import torch
import os.path

from models.NwsResNet import Nws_resnet18
from models.NWS import get_out_in_size, load_codes_from_lzma
import pandas as pd

NUM_CLASSES = (
    200,
    196,
    102,
    195,
    250,
    101
)


def write_csv(root, num_task):
    layer_list = []
    unique_codes_list = []
    counts_list = []
    task_list = []
    for task in range(num_task):
        codes_path = f'{root}/{task}task_codes.lzma'
        model = Nws_resnet18(512, num_classes=NUM_CLASSES[task], gs=1)
        cnn_out_in = get_out_in_size(model)
        codes_layers = load_codes_from_lzma(codes_path, cnn_out_in)

        for layer, codes_layer in enumerate(codes_layers):
            unique_codes, counts = torch.unique(codes_layer, sorted=True, return_counts=True)
            total_count = torch.sum(counts)
            for code, count in zip(unique_codes, counts):
                task_list.append(task+1)
                layer_list.append(layer+1)
                unique_codes_list.append(code.item())
                counts_list.append((count / total_count).item())

    df = pd.DataFrame(data={'task': task_list, 'layer': layer_list, 'codes': unique_codes_list, 'probability': counts_list})
    df.to_csv(os.path.join(root, 'codes_dist.csv'), index=False)


# for others datasets
root = 'H:/tmp_results/cub2sketches_resnet18_lr0.001_e100_nemb512_beta0.1_seed1993'

num_task = 5
write_csv(root, num_task)
## torch.save(cnn_out_in, root + f'cnn_out_in.pt')

## for cifar100
# root = 'H:/new_confirmed/cifar_conti_recon_fullimg160_rt0'
#
# num_task = 20
# model = Nws_resnet18(512, num_classes=5, gs=1)
# meta_data = torch.load(root + f'/{num_task -1 }unique_codes.pt')
# num_emb_layers = meta_data['num_emb_layers']
# cnn_out_in = get_out_in_size(model)
# write_csv(root, num_task)

