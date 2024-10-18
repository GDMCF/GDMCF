pimport configargparse, argparse

def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add('-c', '--config', default='config/ml-1m.yaml', is_config_file=True, help='Config file path')
    
    parser.add_argument('--dataset', type=str, default='ml-1m_clean', help='choose the dataset')
    parser.add_argument('--data_path', type=str, default='../Datasets/yelp_clean/', help='load data path')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
    parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
    parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
    parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
    parser.add_argument('--log_name', type=str, default='log', help='the log name')
    parser.add_argument('--round', type=int, default=1, help='record the experiment') # 
    parser.add_argument('--out_name', type=str, default='GDMCF', help='output name') # 
    parser.add_argument('--debug', type=bool, default=False, help='debug') # 
    parser.add_argument('--noise_type', type=int, default=0, help='continous noise type') # 
    parser.add_argument('--gcnLayerNum', type=int, default=2, help='the number of GCN layer') # 
    parser.add_argument('--user_guided', type=int, default=1, help='user-guided or not') 


    # params for the model
    parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
    parser.add_argument('--dims', type=int, action='append', help='the dims for the DNN')
    parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
    parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')
    parser.add_argument('--backbone', type=str, default='DNN', help='backbone network type')
    parser.add_argument('--OneHotMatrix', type=int, default=1, help='use descrete noise or not')

    # params for diffusion
    parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
    parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale of for continous noise generating')
    parser.add_argument('--noise_min', type=float, default=0.001, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.01, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=25, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
    parser.add_argument('--discrete', type=float, default=0.9995, help='discrete value of diffusion')

    args = parser.parse_args()
    return args
