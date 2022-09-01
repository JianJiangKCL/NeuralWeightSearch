from argparse import ArgumentParser


def project_args(parser: ArgumentParser):
	parser.add_argument("--project", type=str)
	parser.add_argument("--name", type=str)
	parser.add_argument("--wandb_mode", choices=['online', 'offline'], default='offline', type=str)
	parser.add_argument("--lr_logger", choices=[0, 1], default=1, type=int)
	parser.add_argument("--results_dir", default='results', type=str)
	parser.add_argument('-c', '--config_file', type=str, help="xxx.yaml")
	parser.add_argument('-t', '--temporary', action='append', type=str, help="dynamic change args")