#sh run.sh ml-1m_clean 5e-5 0 400 [1000] 10 x0 5 0.0001 0.0005 0.005 0 1 log 1 1 0.1 100
CUDA_VISIBLE_DEVICES=1  python main.py --cuda --config config/ml-1mOneEmbGcn.yaml --sampling_steps 5 --out_name Sampling5FullSeed100 --random_seed 100
CUDA_VISIBLE_DEVICES=1  python main.py --cuda --config config/ml-1mOneEmbGcn.yaml --sampling_steps 5 --out_name Sampling5FullSeed200 --random_seed 200
CUDA_VISIBLE_DEVICES=1  python main.py --cuda --config config/ml-1mOneEmbGcn.yaml --sampling_steps 5 --out_name Sampling5FullSeed300 --random_seed 300
CUDA_VISIBLE_DEVICES=1  python main.py --cuda --config config/ml-1mOneEmbGcn.yaml --sampling_steps 5 --out_name Sampling5FullSeed400 --random_seed 400