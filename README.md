# Neural Weight Search (NWS) 

This is official implementation of [Neural Weight Search for Task Incremental Learning]().

NWS automatically builds new models by searching pretrained grouped weights saved in layer-wise pools and saves these models in the form of indices, which significantly reduces the memory cost.  

NWS is a novel out-of-the-box mechanism that can be easily integrated with modern deep learning methods.

The experiments show NWS achieves state-of-the-art performance on the two task-incremental learning bencharmk, i.e., Split-CIFAR-100 and CUB-to-Sketches benchmarks in terms of both accuracy and memory.


## Prerequisites
Create a new python env named **nws**

Way1: from requirements.txt
```
$ pip install -r requirements.txt

```
Way2: manually install packages
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install pytorch-lightning
pip install einops
pip install pytorch-lightning-bolts
```
## Prepare Benchmark Datasets

Split-Cifar benchmark is available at [here](https://drive.google.com/file/d/1LF45sCaGk33_zTQUdQQaWFDn3ZWQR3RJ/view?usp=sharing)

Cub-to-Sketches benchmark has 5 fine-grained datasets which are available at [here](https://drive.google.com/drive/folders/1Mfdmes07TWhGX-HN90R66lYUzkqbdVhp?usp=sharing)


## Pretrained kernel pools and Experiment Results

   Some of the results are listed in [result_cls.md](./result_cls.md). We provide pretrained kernel pools on Imagenet in [google drive](https://drive.google.com/drive/folders/1Z_q-42zYGB61wngOikyILW9lucA5wppj?usp=sharing)

## Quick Start
Download the ckpt folders or train your own kernel pools 
`configs/xxx.yaml` is the configuration file, which contains network architecture,  NWS-related and training related parameters. 
`scripts/xxx.sh` is the script to run NWS experiments.

* __Split CIFAR-100__ experiments with NWS-incorporated incremental learning
NOTE change the dataset path to your local machine
```
$ bash cifar_resnet18.sh
```

* __CUB to Sketches__ experiments with NWS-incorporated incremental learning

```
$ bash cub2sketches_resnet18.sh
```

## Advanced Usage for Slurm Cluster

`scripts/job_launcher.sh` is the script to submit NWS experiments on slurm cluster.

for example, to submit jobs for `cifar_resnet18.sh` with multiple random seeds:
```
python job_launcher.py -s cifar_res![img.png](img.png)net18.sh -n cifar_resn18 -a 2 
```


## Known Issues

1.When pretaining the kernel pools, if amp16 is used, sometimes Nan loss (due to the overflow) may be induced. 


## Potential Applications
Potentially, the NWS can be applied to other ML domains like model compression, network sparsification. 

We have also designed a toy experiment to show the effectiveness of NWS in network sparsification. The results can be found in the appendix of the paper.

## Citation
If you found the provided code useful, please cite our work.
**Authors**: [Jian Jiang](https://www.linkedin.com/in/jianjiang-kcl/),  [Oya Celiktutan](https://nms.kcl.ac.uk/oya.celiktutan/) from King's College London

if you have any questions, please contact jian.jiang@kcl.ac.uk
```bibtex
Jiang, J., & Celiktutan, O. (2022). Neural Weight Search for Scalable Task Incremental Learning. __arXiv__. https://doi.org/10.48550/arXiv.2211.13823
```
