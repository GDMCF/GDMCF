eval "$(conda shell.bash hook)"
conda activate ddpm
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --config config/yelpOneEmbGcn.yaml # 主程序
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --config config/yelpOneEmbGcn.yaml --out_name WoConNoise --noise_type 1 # 去掉连续噪声，训练容易崩
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --config config/yelpOneEmbGcn.yaml --out_name WoDisNoise --noise_type 2 # 去掉离散噪声 

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --config config/yelpOneEmbGcn.yaml --out_name ns01 --noise_scale 0.05 # 调整连续噪声尺度
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --config config/yelpOneEmbGcn.yaml --out_name ns01 --discrete 0.9995 # 调整离散噪声尺度

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --config config/yelpOneEmbGcn.yaml --out_name Gcnly --gcnLayerNum 1 # Gcn层数（范围：0，1，2）