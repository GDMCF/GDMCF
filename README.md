## GDMCF: A Graph-based Diffusion Model for Collaborative Filtering

## Environment

- Anaconda 3
- Python 3.8.10
- PyTorch 1.12.0
- NumPy 1.22.3

## Data

The experimental data are in the `./Datasets` folder, including Amazon-Book and Yelp. 

## Usage

### Training

```bash
python -u main.py --cuda --dataset=$1 --data_path=./Datasets/$2/ --lr=$3 --weight_decay=$4 --batch_size=$5 --dims=$6 --steps=$7 --noise_scale=$8 --log_name=$9 --gpu=$10 --random_seed $11

```


## Examples

Train GDMCF on Yelp

```bash
python -u main.py --cuda --dataset=yelp_clean --data_path=./Datasets/yelp_clean/ --lr=0.00001 --weight_decay=0.0 --batch_size=400 --dims=[1000] --steps=5 --noise_scale=0.01 --log_name=log --gpu=0 --random_seed=0

```


