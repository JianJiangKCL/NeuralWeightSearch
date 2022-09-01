from argparse import ArgumentParser


def custom_args(parser: ArgumentParser):
	parser.add_argument("--n_emb", type=int, help=' the size of codebook, i.e. the number of embeddings', default=512)
	parser.add_argument("--use_qtz_only", choices=[0, 1], type=int, help='whether to random initial temporary weights', default=0)
	parser.add_argument("--beta", default=0.5, type=float)
	parser.add_argument("--gs", default=1, type=int, help="group size")
	parser.add_argument("--use_recon_codes", choices=[0, 1], type=int, help='whether to use previous reconstructed codes', default=1)
	parser.add_argument("--task_id", default=0, type=int)
	parser.add_argument("--pretrained_end_class", default=1000, type=int)
	parser.add_argument("--end_class", default=5, type=int)