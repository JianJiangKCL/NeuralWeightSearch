from argparse import ArgumentParser


def train_args(parser: ArgumentParser):
	parser.add_argument("--seed", type=int, default=1993)
	parser.add_argument("--arch", type=str, default='resnet18')
	parser.add_argument("--lr", type=float, default=0.1)
	parser.add_argument("--epochs", type=int, default=200)
	parser.add_argument("--warmup_epochs", type=int, default=10)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--optimizer", default='sgd', type=str)
	parser.add_argument("--scheduler", default='multistep', type=str)
	parser.add_argument("--scheduler_interval", default=None, type=str)
	parser.add_argument("--disable_tqdm", action='store_true', help="disable tqdm progress bar")
	parser.add_argument("--distributed", action="store_true", help="distribute over different gpus ")
	parser.add_argument("--gpus", type=int, default=1)
	parser.add_argument('--resume', choices=[0, 1], default=0, type=int)
	parser.add_argument('--finetune', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
	parser.add_argument("--use_amp", choices=[0, 1], default=1, type=int, help="use apex for mixed precision training")
	parser.add_argument("--use_swa", choices=[0, 1], default=0, type=int, help="use Stochastic Weight Averaging")
	parser.add_argument("--grad_clip", choices=[0, 1], default=0, type=int, help="use gradient clipping")