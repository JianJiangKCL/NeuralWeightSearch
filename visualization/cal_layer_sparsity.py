import torch
import os.path

from models.NwsResNet import Nws_resnet18
from models.NWS import get_out_in_size, load_codes_from_lzma
import numpy as np
import pandas as pd

NUM_CLASSES=(
    200,
    196,
    102,
    195,
    250,
    101
)


def write_csv(root, num_task):
	layer_list = []
	sparsity_list = []
	utilization_list = []
	task_list = []
	for task in range(num_task):
		codes_path = f'{root}/{task}task_codes.lzma'
		model = Nws_resnet18(512, num_classes=NUM_CLASSES[task], gs=1)
		cnn_out_in = get_out_in_size(model)
		codes_layers = load_codes_from_lzma(codes_path, cnn_out_in)

		for layer, codes_layer in enumerate(codes_layers):
			unique_codes, counts = torch.unique(codes_layer, sorted=True, return_counts=True)
			num_unique_codes = len(unique_codes)
			total_kernels = cnn_out_in[layer][0] * cnn_out_in[layer][1]
			threshold = np.sqrt(total_kernels)
			# less_used_counts = counts[counts < threshold]
			dominant_k_counts = counts[counts > threshold]
			summed_dominant_k_counts = torch.sum(dominant_k_counts)

			sparsity = 1 - summed_dominant_k_counts / total_kernels
			utilization = num_unique_codes / 512
			task_list.append(task+1)
			layer_list.append(layer+1)
			sparsity_list.append(sparsity.item())
			utilization_list.append(utilization)
	df = pd.DataFrame(data={'task': task_list, 'layer': layer_list, 'utilization ratio': utilization_list, 'sparsity': sparsity_list})
	df.to_csv(os.path.join(root, 'layer_sparsity.csv'), index=False)



# root = 'H:/tmp_results/confirmed/Nws_ret/shrink_others_fullimgckpt160_recon/'
root = 'H:/tmp_results/cub2sketches_resnet18_lr0.001_e100_nemb512_beta0.1_seed1993'
num_task = 5
# model = Nws_resnet18(512, num_classes=NUM_CLASSES[1], gs=1)
# # meta_data = torch.load(root + f'/{num_task -1 }unique_codes.pt')
# # num_emb_layers = meta_data['num_emb_layers']
# cnn_out_in = get_out_in_size(model)
write_csv(root, num_task)
# torch.save(cnn_out_in, root + f'cnn_out_in.pt')

