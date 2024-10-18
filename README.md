## GDMCF: A Graph-based Diffusion Model for Collaborative Filtering

## Environment

- Anaconda 3
- Python 3.8.10
- PyTorch 1.12.0
- NumPy 1.22.3

## Data

The experimental data are in the `./datasets` folder, including Amazon-Book and Yelp. 

Note that the results on ML-1M differ from those reported in CODIGEM, owing to different data processing procedures. CODIGEM did not sort and split the training/testing sets according to timestamps; however, temporal splitting aligns better with real-world testing.

## Usage

### Training

```bash
python -u main.py --cuda --dataset=$1 --data_path=./Datasets/$2/ --lr=$3 --weight_decay=$4 --batch_size=$5 --dims=$6 --steps=$7 --noise_scale=$8 --log_name=$9 --round=$10 --gpu=$11 --discrete $12 --random_seed $13

```


### Inference

```bash
CUDA_VISIBLE_DEVICES=$1 python inference.py --dataset=$2 

```

## Examples

1. Train GDMCF on Yelp

```bash
python -u main.py --cuda --dataset=yelp_clean --data_path=./datasets/yelp_clean/ --lr=0.00001 --weight_decay=0.0 --batch_size=400 --dims=[1000] --steps=5 --noise_scale=0.01 --log_name=log --gpu=0 --discrete=0.99 --random_seed=0



```

2. Inference GDMCF on Yelp
   
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --dataset=yelp_clean

```

