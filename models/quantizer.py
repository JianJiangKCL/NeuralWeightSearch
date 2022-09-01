import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantize(nn.Module):
	def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
		super().__init__()

		self.dim = dim
		self.n_embed = n_embed
		self.decay = decay
		self.eps = eps

		# dxN_e
		embed = torch.randn(dim, n_embed)
		self.register_buffer("embed", embed)
		self.register_buffer("cluster_size", torch.zeros(n_embed))
		self.register_buffer("embed_avg", embed.clone())

		self.register_buffer("acum_embed_sum", torch.zeros_like(self.embed))
		self.register_buffer("acum_embed_onehot_sum", torch.zeros(n_embed))

	def zero_buffer(self):

		self.acum_embed_onehot_sum.data = torch.zeros_like(self.acum_embed_onehot_sum)
		self.acum_embed_sum.data = torch.zeros_like(self.embed)

	def update(self):

		self.cluster_size.data.mul_(self.decay).add_(
			self.acum_embed_onehot_sum, alpha=1 - self.decay
		)

		self.embed_avg.data.mul_(self.decay).add_(self.acum_embed_sum, alpha=1 - self.decay)
		n = self.cluster_size.sum()
		cluster_size = (
				(self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
		)
		embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
		self.embed.data.copy_(embed_normalized)

	def update_new_emb(self):
		n_emb = self.n_embed
		# [n_emb:] means the new embs
		self.cluster_size.data[n_emb:].mul_(self.decay).add_(
			self.acum_embed_onehot_sum[n_emb:], alpha=1 - self.decay
		)

		self.embed_avg.data[:, n_emb:].mul_(self.decay).add_(self.acum_embed_sum[:, n_emb:], alpha=1 - self.decay)

		n = self.cluster_size[n_emb:].sum()
		cluster_size = (self.cluster_size[n_emb:] + self.eps) / (n + n_emb * self.eps) * n

		embed_normalized = self.embed_avg[:, n_emb:] / cluster_size.unsqueeze(0)
		self.embed.data[:, n_emb:].copy_(embed_normalized)

	def forward(self, input):
		# for weights, input is out_c, in_c, h*w
		# out_dim, in_dim
		flatten = input.reshape(-1, self.dim)

		# @timeit
		def cal_dist1():
			dist = torch.cdist(flatten, self.embed.permute(1, 0), compute_mode="use_mm_for_euclid_dist")
			return dist

		dist = cal_dist1()
		_, embed_ind = (-dist).max(1)
		embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)

		# permute input is necessary.
		embed_ind = embed_ind.view(*input.shape[:-1])
		quantize = self.embed_code(embed_ind)

		if self.training:
			# Note treat B*H*W as positions; C is a set of values in a certain position
			# B*H*WxN_e => N_e; it means how many times an embedding is chosen during this mini-batch
			embed_onehot_sum = embed_onehot.sum(0)
			# Note an embedding's value is learnt from its members' vector values (the avg of members' value)
			#   CxB*H*W matmul* B*H*WxN_e => CxN_e ;
			#
			#  the summation of corresponding dim from selected members for an embedding;
			embed_sum = flatten.transpose(0, 1).contiguous() @ embed_onehot
			####
			# with torch.no_grad():
			self.acum_embed_onehot_sum.data.add_(embed_onehot_sum)
			self.acum_embed_sum.data.add_(embed_sum)

		# stop gradients to embeddings; inputs are updated by diff to get close to corresponding embedding
		diff = (quantize.detach() - input).pow(2).mean()
		# this brings the gradients induced by quantization and following layers to the input, i.e. temporary weights.
		# below is straight-through estimation for gradients
		quantize = input + (quantize - input).detach()
		return quantize, diff, embed_ind

	def reset_nemb(self, nemb):
		self.n_embed = nemb
		self.embed = torch.randn(self.dim, nemb)
		self.embed_avg = self.embed.clone()
		self.cluster_size = torch.zeros(nemb).cuda()
		self.acum_embed_sum = torch.zeros([self.dim, nemb]).cuda()
		self.acum_embed_onehot_sum = torch.zeros(nemb).cuda()

	def embed_code(self, embed_id):
		return F.embedding(embed_id, self.embed.transpose(0, 1).contiguous())

	def embed_sparse_code(self, embed_id, sparse_codes):
		# [ n, dim]
		embed = self.embed.transpose(0, 1).contiguous()
		zeros = torch.zeros(sparse_codes.size(0), embed.size(1)).cuda()
		embed[sparse_codes, :] = zeros
		return F.embedding(embed_id, embed)

	def embed_sparse_code_high(self, embed_id, sparse_codes):
		# [ n, dim]
		embed = self.embed.transpose(0, 1).contiguous()
		zeros = torch.zeros(sparse_codes.size(0), embed.size(1)).cuda()
		embed[sparse_codes, :] = zeros
		return F.embedding(embed_id, embed)

	def encode(self, input):
		# out_dim, in_dim
		flatten = input.reshape(-1, self.dim)

		def cal_dist1():
			dist = torch.cdist(flatten, self.embed.permute(1, 0), compute_mode="use_mm_for_euclid_dist")
			return dist

		dist = cal_dist1()

		_, embed_ind = (-dist).max(1)

		embed_ind = embed_ind.view(*input.shape[:-1])
		return embed_ind

	def embed_code_straight_through(self, input, embed_id):
		quantize = F.embedding(embed_id, self.embed.transpose(0, 1).contiguous())
		quantize = input + (quantize - input).detach()
		return quantize


def timeit(f):
	def timed(*args, **kw):
		ts = time.time()
		result = f(*args, **kw)
		te = time.time()
		print('Function "{name}" took {time} seconds to complete.'.format(name=f.__name__, time=te - ts))
		return result

	return timed
