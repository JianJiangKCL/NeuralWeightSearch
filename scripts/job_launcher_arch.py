import sys
import os
import subprocess
import argparse
from datetime import datetime
import inspect
import random
import string

# printing lowercase
letters = string.ascii_lowercase
suffix = ''.join(random.choice(letters) for i in range(6))


parser = argparse.ArgumentParser()
parser.add_argument("--script", '-s', type=str, required=True)
parser.add_argument("--mode", type=str, choices=['slurm', 'normal'], default="slurm")
parser.add_argument("--name", '-n', type=str, default=None)
parser.add_argument("--root", type=str, default="commands")
parser.add_argument("--num_gpus", '-gpu', type=int, default=1)
parser.add_argument("--arrays", '-a', type=int, default=0)
parser.add_argument("--hours", type=int, default=72)
parser.add_argument("--arch", type=str)
args = parser.parse_args()


# CKPT_PATHS={
#     'mobilnetv2':'/jmain02/home/J2AD019/exk04/jxj51-exk04/proj/KSD/results/subimagenet_mv22/subimagenet_mobilenetv2/lr0',
#     'vgg16':''
#
# }
if args.num_gpus > 1:
    args.hours = 24

else:
    args.hours = 96

# load file
if os.path.exists(args.script):
    with open(args.script) as f:
        command = f.read()

else:
    print(f"{args.script} does not exist.")
    exit()
command_dir = args.root
os.makedirs(command_dir, exist_ok=True)
print(f"temporary commands directory: {command_dir}")

use_ddp = 1 if args.num_gpus > 1 else 0
if args.arch == 'mobilenetv2':
    ckpt_path = 'results/mobilenet_v2-b0353104.pth'
elif args.arch == 'vgg16':
    ckpt_path = 'results/vgg16_bn-6c64b313.pth'
else:
    raise ValueError('architecture not supported')
# run command
total_cpus = 2 * args.num_gpus
if args.mode == "normal":
    # p = subprocess.Popen(f"bash {args.script} {args.name} {use_ddp}", shell=True, stdout=sys.stdout, stderr=sys.stdout)
    # p.wait()
    cmd = f"bash {args.script} {args.arch} {ckpt_path}"
    print('======', cmd)
    os.system(cmd)

elif args.mode == "slurm":

    # build slurm command
    command_prefix = inspect.cleandoc(
        f"""
        #!/bin/bash
        #SBATCH --job-name {args.arch}     
        #SBATCH --nodes=1
        #SBATCH --gres gpu:{args.num_gpus}
        #SBATCH --cpus-per-task {total_cpus}
  
        #SBATCH --time {args.hours}:00:00
    
        
        # mail alert at start, end and abortion of execution
        #SBATCH --mail-type=ALL
        
        # send mail to this address
        #SBATCH --mail-user=jian.jiang@kcl.ac.uk
      

        # loading conda env
        source ~/.bashrc
        conda activate nws
        
        # the rest is the scripts

        """
    )

    # write temporary command
    command_path = os.path.join(command_dir, f"command_{suffix}.sh" )
    command = command_prefix + command
    with open(command_path, "w") as f:
        f.write(command)

    sbatch_type = None
    if args.num_gpus == 1:
        sbatch_type = f'sbatch -p small --gres=gpu:1 --cpus-per-task={total_cpus}'
    elif args.num_gpus == 4:
        sbatch_type = f'sbatch -p big --gres=gpu:4 --cpus-per-task={total_cpus}'
    elif args.num_gpus == 8:
        sbatch_type = f'sbatch -p big --gres=gpu:8 --cpus-per-task={total_cpus}'

    if args.arrays != 0:
        sbatch_type += f' --array=0-{args.arrays}'
    # run command
    bash_command = f"bash {command_path}  {args.arch}"
    print(f"running command: {bash_command}")
    # p = subprocess.Popen(bash_command, shell=True, stdout=sys.stdout, stderr=sys.stdout)

    p = subprocess.Popen(f"{sbatch_type} {command_path} {args.arch} {ckpt_path}", shell=True, stdout=sys.stdout)#, stderr=sys.stdout)
    p.wait()
