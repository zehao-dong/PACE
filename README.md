PACE: Parallelizable Attention-based Computation Structure Encoder for Directed Acyclic Graphs
===============================================================================

Installation
------------

Tested with Pyorch==1.8.1, torchvision==0.9.1

Install python-igraph by:
    pip install python-igraph

Install pygraphviz by:
    conda install graphviz
    conda install pygraphviz

Other required python libraries: 
tqdm,
six, 
scipy, 
numpy, 
matplotlib, 
hop,
click,
Cython,
ipython, 
pandas,
pathvalidate,
scipy,
seaborn,
statsmodels,
psutil,
scikit-image,
autograd>=1.3,
emcee==3.0.2,
lightgbm>=2.3.1,
networkx==2.2
Pillow>=7.1.2,
transformers==4.6.1, 
ConfigSpace==0.4.18,
scikit-learn>=0.23.1,
tensorboard==1.15.0,
tensorflow-gpu==1.15.0,
tensorflow-estimator,


NA and BN
--------

#### Training NA

###### Training PACE
python train_pace_na.py --data-name final_structures6 --save-interval 25 --save-appendix _PACE --epochs 100 --lr 1e-4 --cuda_number 0 --batch-size 32 --ninp 32 --nhid 64 --dropout 0.15 --nlayers 3 --nz 64

###### Training PACE + no mask
python train_pace_na_nomask.py --data-name final_structures6 --save-interval 50 --save-appendix _PACEnomask --epochs 100 --lr 1e-4 --cuda_number 0 --batch-size 32 --ninp 32 --nhid 64 --dropout 0.15 --nlayers 3 --nz 64

###### Training PACE + no dag2seq
python train_pace_na_nodagseq.py --data-name final_structures6 --save-interval 50 --save-appendix _PACEnodagseq --epochs 100 --lr 1e-4 --cuda_number 0 ---batch-size 32 --ninp 32 --nhid 64 --dropout 0.15 --nlayers 3 --nz 64


#### Training BN

###### Training PACE
python train_pace_bn.py --data-name asia_200k --save-interval 10 --save-appendix _DTRANSVAE --epochs 50 --lr 1e-4 --cuda_number 1 --batch-size 32 --ninp 32 --nhid 64 --dropout 0.25 --nlayers 3 --data-type BN --nvt 8 --nz 64

###### Training PACE + no mask
python train_pace_bn_mask.py --data-name asia_200k  --save-interval 10 --save-appendix _PACEmask --epochs 50 --lr 1e-4 --cuda_number 0  --batch-size 32 --ninp 32 --nhid 64 --dropout 0.25 --nlayers 3 --data-type BN --nvt 8 --nz 64

###### Training PACE + no dag2seq
python train_pace_bn_nodagseq.py --data-name asia_200k  --save-interval 10 --save-appendix _PACEnodagseq --epochs 50 --lr 1e-4 --cuda_number 1 --batch-size 32 --ninp 32 --nhid 64 --dropout 0.25 --nlayers 3 --data-type BN --nvt 8 --nz 64

Settings for baselines are available at: https://github.com/muhanzhang/D-VAE

#### Bayesian Optimization

To perform Bayesian optimization experiments after training PACE, the following additional steps are needed.

Install sparse Gaussian Process (SGP) based on Theano:

    cd bayesian_optimization/Theano-master/
    python setup.py install
    cd ../..

Download the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) by: 

    cd software/enas
    mkdir data
    cd data
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xzvf cifar-10-python.tar.gz
    mv cifar-10-batches-py/ cifar10/
    cd ../..

Download the 6-layer [pretrained ENAS model](https://drive.google.com/drive/folders/1e-mYRZS_10Aegj8Sczcb948RbHyiju1S?usp=sharing) to "software/enas/" (for evaluating a neural architecture's weight-sharing accuracy). There should be a folder named "software/enas/outputs_6/", which contains four model files. The 12-layer pretrained ENAS model is available [here](https://drive.google.com/drive/folders/18GU9g5DNiHn2MOVKOiF1fCwNQMTA-mnH?usp=sharing) too.

Install [TensorFlow](https://www.tensorflow.org/install/gpu) >= 1.12.0

Install R package _bnlearn_:

    R
    install.packages('bnlearn', lib='/R/library', repos='http://cran.us.r-project.org')


Then, in "bayesian_optimization/":

###### BO on NA
./run_bo_ENAS.sh
./run_bo_ENAS_nomask.sh
./run_bo_ENAS_nodagseq.sh

###### BO on BN
./run_bo_BN.sh
./run_bo_BN_nomask.sh
./run_bo_BN_nodagseq.sh


NAS101 and NAS301
--------

### Generate training data (for MLM objective)
python preprocessing/gen_json.py
python preprocessing/data_generate.py --dataset nasbench101 --flag extract_seq
python preprocessing/data_generate.py --dataset nasbench101 --flag build_pair --k 2 --d 2000000 --metric params
python preprocessing/gen_json_darts.py 
python preprocessing/data_generate.py --dataset nasbench301 --flag extract_seq
python preprocessing/data_generate.py --dataset nasbench301 --flag build_pair --k 1 --d 5000000 --metric flops

### Pre-training NAS101

###### Pre-training PACE
bash scripts/pretrain_nasbench101_pace.sh
###### Pre-training PACE + no mask
bash scripts/pretrain_nasbench101_pace_nomask.sh
###### Pre-training PACE + no dag2seq
bash scripts/pretrain_nasbench101_pace_nodagseq.sh


### Pre-training NAS301
Settings for preprocessing process and pre-training process are available at https://github.com/MSU-MLSys-Lab/CATE

###### Pre-training PACE
bash scripts/pretrain_nasbench301_pace.sh
###### Pre-training PACE + no mask
bash scripts/pretrain_nasbench301_pace_nomask.sh
###### Pre-training PACE + no dag2seq
bash scripts/pretrain_nasbench301_pace_nodagseq.sh

Settings for all baselines are available at: https://github.com/muhanzhang/D-VAE and https://github.com/vthost/DAGNN
or (examples):
python inference/dvae.py  --train_data data/nasbench101/train_data.pt --valid_data data/nasbench101/test_data.pt --dataset nasbench101
python inference/dvae.py  --train_data data/nasbench301/train_data.pt --valid_data data/nasbench301/test_data.pt --dataset nasbench301
python inference/run_dagnn.py  --train_data data/nasbench101/train_data.pt --valid_data data/nasbench101/test_data.pt --dataset nasbench101
python inference/run_dagnn.py  --train_data data/nasbench301/train_data.pt --valid_data data/nasbench301/test_data.pt --dataset nasbench301


### Downstram search on NAS101

First, we need to generate the DAG encodings based on trained DAG encoders

PACE:

python inference/pace_inference.py --pretrained_path pace/nasbench101_model_best.pth.tar --train_data data/nasbench101/train_data.pt --valid_data data/nasbench101/test_data.pt --dataset nasbench101 --weight_scale 0.01

PACE + no mask:

python inference/pace_inference_nomask.py --pretrained_path pace_nomask/nasbench101_model_best.pth.tar --train_data data/nasbench101/train_data.pt --valid_data data/nasbench101/test_data.pt --dataset nasbench101

PACE + no dag2seq:

python inference/pace_inference_nodagseq.py --pretrained_path pace_nodagseq/nasbench101_model_best.pth.tar --train_data data/nasbench101/train_data.pt --valid_data data/nasbench101/test_data.pt --dataset nasbench101

Next, we can run Bayesian optimization DNGO and DNGO-LS

###### DNGO

PACE:
bash scripts/nas101/run_search_nasbench101_pace.sh

PACE + no mask:
bash scripts/nas101/run_search_nasbench101_nomask.sh

PACE + no dag2seq
bash scripts/nas101/run_search_nasbench101_nodagseq.sh

baseline examples:
bash scripts/nas101/run_search_nasbench101_dagnn.sh
bash scripts/nas101/run_search_nasbench101_dvae.sh

###### DNGO-LS

PACE:
bash scripts/nas101/run_search_ls_nasbench101_pace.sh

PACE + no mask:
bash scripts/nas101/run_search_nasbench101_ls_nomask.sh

PACE + no dag2seq
bash scripts/nas101/run_search_nasbench101_ls_nodagseq.sh

baseline examples:
bash scripts/nas101/run_search_ls_nasbench101_dagnn.sh
bash scripts/nas101/run_search_ls_nasbench101_dvae.sh


### Downstram search on NAS301

First, we need to generate the DAG encodings based on trained DAG encoders

PACE:
python inference/pace_inference.py --pretrained_path pace/nasbench301_model_best.pth.tar --train_data data/nasbench301/train_data.pt --valid_data data/nasbench301/test_data.pt --dataset nasbench301 --n_vocab 11 --weight_scale 0.1

PACE + no mask:
python inference/pace_inference_nomask.py --pretrained_path pace_nomask/nasbench301_model_best.pth.tar --train_data data/nasbench301/train_data.pt --valid_data data/nasbench301/test_data.pt --dataset nasbench301 --n_vocab 11

PACE + no dag2seq:
python inference/pace_inference_nodagseq301.py --pretrained_path pace_nodagseq/nasbench301_model_best.pth.tar --train_data data/nasbench301/train_data.pt --valid_data data/nasbench301/test_data.pt --dataset nasbench301 --n_vocab 11


Next, we can run Bayesian optimization DNGO and DNGO-LS

###### DNGO

PACE:
bash scripts/nas301/run_search_nasbench301_pace.sh

PACE + no mask:
bash scripts/nas301/run_search_nasbench301_nomask.sh

PACE + no dag2seq
bash scripts/nas301/run_search_nasbench301_nodagseq.sh

baseline examples:
bash scripts/nas301/run_search_nasbench301_dagnn.sh
bash scripts/nas301/run_search_nasbench301_dvae.sh

###### DNGO-LS

PACE:
bash scripts/nas301/run_search_ls_nasbench301_pace.sh

PACE + no mask:
bash scripts/nas301/run_search_nasbench301_ls_nomask.sh

PACE + no dag2seq
bash scripts/nas301/run_search_nasbench301_ls_nodagseq.sh

baseline examples:
bash scripts/nas301/run_search_ls_nasbench301_dagnn.sh
bash scripts/nas301/run_search_ls_nasbench301_dvae.sh
 
 
