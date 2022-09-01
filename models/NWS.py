import lzma
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from numpy import prod
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from .quantizer import Quantize


class NwsConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, gs=1):
		super(NwsConv, self).__init__()

		# self.d_emb = kernel_size **2
		self.kernel_size = kernel_size
		self.weight = Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size)))
		torch.nn.init.kaiming_normal(self.weight)
		self.stride = stride
		self.padding = padding
		self.gs = gs
		self.diff = 0

	def forward(self, x, quantizer):  # ):#
		tofind_c = rearrange(self.weight,
		                     ' o (split i) h w -> o i (split h w)', split=self.gs, h=self.kernel_size, w=self.kernel_size)

		found_c, diff, ids = quantizer(tofind_c)

		found_c = rearrange(found_c, 'o i (split h w) -> o (split i) h w', split=self.gs, h=self.kernel_size, w=self.kernel_size)

		self.diff = diff

		out = F.conv2d(x, found_c, None, self.stride, self.padding)

		return out

	def straight_forward(self, x):
		out = F.conv2d(x, self.weight, None, self.stride, self.padding)
		return out

	def encode_weights(self):
		tobuy_c = rearrange(self.weight, ' o (split i) h w -> o i (split h w)', split=self.gs, h=self.kernel_size, w=self.kernel_size)
		embed_ind = self.qtz.encode(tobuy_c)
		embed_ind = embed_ind.long().detach().cpu().flatten()
		return embed_ind


class Nwsxxx(NwsConv):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, gs=1):
		super(Nwsxxx, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, gs=gs)

		self.qtz = None
		self.eval_mode = False

	def forward(self, x):
		if self.eval_mode:
			out = super().straight_forward(x)
		else:
			out = super().forward(x, self.qtz)
		return out

	def set_eval(self, eval_mode):
		self.eval_mode = eval_mode


class Nws3x3(Nwsxxx):
	def __init__(self, n_emb, in_channels, out_channels, stride=1, padding=0, gs=1):
		super(Nws3x3, self).__init__(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, gs=gs)

		self.qtz = Quantize(9 * gs, n_emb)


class Nws1x1(Nwsxxx):
	# for VGG gs size should be 128, otherwise computation too much
	def __init__(
			self, n_emb, in_channels, out_channels, stride=1, padding=0, gs=1):
		super(Nws1x1, self).__init__(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, gs=gs)

		self.qtz = Quantize(1 * gs, n_emb)


class Nws7x7(Nwsxxx):
	def __init__(
			self, n_emb, in_channels, out_channels, stride=2, padding=3, gs=1):
		super(Nws7x7, self).__init__(in_channels, out_channels, kernel_size=7, stride=stride, padding=padding, gs=gs)

		self.qtz = Quantize(49 * gs, n_emb)


def load_qtzs(ckpt_path, model):
	qtzs = torch.load(f"{ckpt_path}/qtzs")
	cnt = 0
	for n, module in model.named_modules():
		if isinstance(module, NwsConv):
			block = eval('model.' + n[:-5])
			setattr(block, n[-4:], qtzs[cnt])
			cnt += 1


def load_codes(ckpt_path, model):
	qtzs = torch.load(f"{ckpt_path}/qtzs")
	bns = torch.load(f"{ckpt_path}/bns")
	cnn_out_in = torch.load(f"{ckpt_path}/out_in")

	codes_name = f"{ckpt_path}/model_codes_lzma"
	obj = lzma.LZMAFile(codes_name, mode="rb")

	bits_data = obj.read()
	decoded = lzma.decompress(bits_data)
	back_data = np.frombuffer(decoded, dtype=np.int16)
	model_size = 0
	for item in cnn_out_in:
		t = [*item][:2]
		model_size += prod(t)

	flattened_codes = back_data
	offset_layer = 0
	cnt = 0
	cnt_bn = 0
	for module in model.modules():
		if isinstance(module, NwsConv):
			# print(cnt)
			qtzs[cnt].eval()
			out_in = cnn_out_in[cnt]
			length = prod([*out_in][:2])
			tmp = flattened_codes[offset_layer: offset_layer + length]
			# codes = rearrange(tmp, '(1 oi) ->1 oi ')
			codes = torch.from_numpy(tmp).long()  # .cuda()
			weights = qtzs[cnt].embed_code(codes)  # .cpu()
			o_i, h_w = [*weights.size()]
			if h_w == 128:
				h, w = 1, 1
				weights = rearrange(weights, '(o i) (h w)-> o i h w', o=out_in[0], i=out_in[1], h=h, w=w)
			else:
				h = w = int(sqrt(h_w))
				weights = rearrange(weights, '(o i) (h w)-> o i h w', o=out_in[0], i=out_in[1], h=h, w=w)
			module.weight.data = weights
			offset_layer += length

			cnt += 1
		elif isinstance(module, nn.BatchNorm2d):
			# even deepcopy cannot influence it
			module.running_mean = bns[cnt_bn].running_mean
			module.running_var = bns[cnt_bn].running_var
			module.weight = bns[cnt_bn].weight
			module.bias = bns[cnt_bn].bias
			cnt_bn += 1


def map_codes(codes, num_emb):
	embs_indices = torch.arange(0, num_emb)
	unique_indices = torch.unique(codes, sorted=True)
	mapping_list = torch.ones_like(embs_indices) * -1
	for i in range(len(unique_indices)):
		mapping_list[unique_indices[i]] = i
	mapped_indices = mapping_list[codes]
	return mapped_indices, mapping_list


def get_out_in_size(model):
	out_in = []
	for module in model.modules():
		if isinstance(module, NwsConv):
			out_in.append(module.weight.size())

	return out_in


def load_codes2weights_individual_(model, model_codes_layers, cnn_out_in, qtzs, bns):
	cnt = 0
	cnt_bn = 0
	for module in model.modules():
		if isinstance(module, NwsConv):

			codes = model_codes_layers[cnt].cuda()
			qtzs[cnt].eval().cuda()
			weights = qtzs[cnt].embed_code(codes)  # .cpu()
			# o_i, h_w = [*weights.size()]
			o, i, h, w = [*cnn_out_in[cnt]]
			# h = w = int(sqrt(h_w))
			weights = rearrange(weights, '(o i) (h w)-> o i h w', o=o, i=i, h=h, w=w)
			module.weight.data = weights
			cnt += 1
		elif isinstance(module, nn.BatchNorm2d):
			# even deepcopy cannot influence it
			module.running_mean = bns[cnt_bn].running_mean
			module.running_var = bns[cnt_bn].running_var
			module.weight = bns[cnt_bn].weight
			module.bias = bns[cnt_bn].bias
			cnt_bn += 1


def load_codes2weights_sparsified_(model, model_codes_layers, sparse_codes_layers, cnn_out_in, qtzs, bns):
	cnt = 0
	cnt_bn = 0
	for module in model.modules():
		if isinstance(module, NwsConv):
			# print(f'layer {cnt}')
			# if cnt == 2:
			#     k=1
			codes = model_codes_layers[cnt].cuda()
			qtzs[cnt].eval().cuda()
			weights = qtzs[cnt].embed_sparse_code(codes, sparse_codes_layers[cnt])

			o, i, h, w = [*cnn_out_in[cnt]]
			# h = w = int(sqrt(h_w))
			weights = rearrange(weights, '(o i) (h w)-> o i h w', o=o, i=i, h=h, w=w)
			module.weight.data = weights
			cnt += 1
		elif isinstance(module, nn.BatchNorm2d):
			# even deepcopy cannot influence it
			module.running_mean = bns[cnt_bn].running_mean
			module.running_var = bns[cnt_bn].running_var
			module.weight = bns[cnt_bn].weight
			module.bias = bns[cnt_bn].bias
			cnt_bn += 1


def load_codes2weights_sparsified_high_(model, sparse_codes_layers, cnn_out_in, qtzs, bns):
	cnt = 0
	cnt_bn = 0
	for module in model.modules():
		if isinstance(module, NwsConv):

			codes = sparse_codes_layers[cnt].cuda()
			qtzs[cnt].eval().cuda()
			qtzs[cnt].add_zero_emb('cuda')
			weights = qtzs[cnt].embed_code(codes)

			o, i, h, w = [*cnn_out_in[cnt]]
			weights = rearrange(weights, '(o i) (h w)-> o i h w', o=o, i=i, h=h, w=w)
			module.weight.data = weights
			cnt += 1
		elif isinstance(module, nn.BatchNorm2d):
			# even deepcopy cannot influence it
			module.running_mean = bns[cnt_bn].running_mean
			module.running_var = bns[cnt_bn].running_var
			module.weight = bns[cnt_bn].weight
			module.bias = bns[cnt_bn].bias
			cnt_bn += 1


def get_mapping_list_layers(unique_codes_layers, num_emb_layers):
	mapped_unique_codes_layers = []
	mapping_list_layers = []
	for cnt, unique_codes in enumerate(unique_codes_layers):
		embs_indices = torch.arange(0, num_emb_layers[cnt])
		mapping_list = torch.ones_like(embs_indices) * -1
		for i in range(len(unique_codes)):
			mapping_list[unique_codes[i]] = i
		mapped_unique_codes_layers.append(mapping_list[unique_codes])
		mapping_list_layers.append(mapping_list)
	for codes in mapped_unique_codes_layers:
		flag = len((codes == torch.LongTensor([-1])).nonzero())
		assert flag == 0
	return mapped_unique_codes_layers, mapping_list_layers


def delete_unused_embs_(qtzs, unique_codes_layers):
	num_emb_layers = []
	for cnt, unique_codes in enumerate(unique_codes_layers):
		qtzs[cnt].keep_emb_and_delete_others(unique_codes)
		num_emb_layers.append(qtzs[cnt].n_embed)
		k = 1
	return num_emb_layers


def make_codes_unflatten(flattened_codes, cnn_out_in):
	model_size = 0
	for item in cnn_out_in:
		t = [*item][:2]
		model_size += prod(t)
	offset_layer = 0
	codes_layers = []
	for cnt, out_in in enumerate(cnn_out_in):
		length = prod([*out_in][:2])
		codes = flattened_codes[offset_layer: offset_layer + length]
		# codes = rearrange(tmp, '(1 oi) ->1 oi ')
		if isinstance(codes, np.ndarray):
			codes = torch.from_numpy(codes)
		codes = codes.long()
		codes_layers.append(codes)
		offset_layer += length
	return codes_layers


def make_codes_flatten(codes_layers):
	flattened_codes = torch.LongTensor([])
	for codes in codes_layers:
		flattened_codes = torch.cat((flattened_codes, codes.flatten()), dim=0)
	return flattened_codes


def load_codes_from_lzma(codes_path, cnn_out_in):
	obj = lzma.LZMAFile(codes_path, mode="rb")
	bits_data = obj.read()
	decoded = lzma.decompress(bits_data)
	flattened_codes = np.frombuffer(decoded, dtype=np.int16)
	codes_layers = make_codes_unflatten(flattened_codes, cnn_out_in)
	return codes_layers


def save_model_codes(codes, file_name):
	codes = codes.numpy().astype(np.int16)
	# np.savez(file_name, model_codes=codes)
	with lzma.open(file_name + '.lzma', 'wb') as f:
		f.write(lzma.compress(codes))