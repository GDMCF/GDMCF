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
CUDA_VISIBLE_DEVICES=$1 python main.py --cuda --config config/$2 --out_name $3 --noise_scale $4 --discrete $5 --gcnLayerNum $6

```


### Inference

```bash
CUDA_VISIBLE_DEVICES=$1 python inference.py --dataset=$2 

```

## Examples

1. Train GDMCF on Amazon-book

```bash
CUDA_VISIBLE_DEVICES=$1 python main.py --cuda --config config/$2 --out_name $3 --noise_scale $4 --discrete $5 --gcnLayerNum $6

```

2. Inference GDMCF on Yelp
   
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --dataset=yelp

```

