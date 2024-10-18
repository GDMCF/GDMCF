"""
Train a diffusion model for recommendation
"""

import argparse
from ast import parse
import os, sys
import time
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import models.gaussian_diffusion as gd
from models.DNN import DNN, DNNCat, DNNCat2, DNNOneHot, DNNlightGCN, DNNOneHotTransformer, DNNOneHotEmbedding, DNNOneHotEmbeddingGCN, DNNOneHotEmbedding_conti
from models.DNN import DNNOneHotEmbeddingGCN_conti
import evaluate_utils
import data_utils
from copy import deepcopy
from parse_args_util import parse_args
from data_utils import adjacency_to_edge, edge_to_adjacency
import random
from datetime import datetime
random_seed = 1
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
def adjacency_to_one_hot(a, b, X):
    """
    将形状为[a, b]的邻接矩阵X转换为形状为[a+b, a+b]的one-hot表示形式。
    
    参数:
    - a: 节点数
    - b: 可能的边数（假设是无向图，所以b应该是a*(a-1)/2，但这里直接使用b作为输入）
    - X: 形状为[a, b]的二值邻接矩阵
    
    返回:
    - Y: 形状为[a+b, a+b]的one-hot编码矩阵
    """
    # 初始化新矩阵为0
    Y = torch.zeros((a + b, a + b))
    
    # 由于是无向图，我们需要同时考虑i到j和j到i的情况，但one-hot编码通常不用于直接表示这种关系
    # 这里我们简化处理，直接基于X的结构构造Y，确保每条边只被表示一次，并符合问题描述的意图
    # for i in range(a):  # 遍历所有可能的起始节点
    #     for j in range(b):  # 遍历所有可能的目标"边"索引，需转换理解
    #         if X[i, j] == 1:  # 如果存在连接
    #             # 注意这里的转换逻辑需根据实际情况调整，以下仅为示意
    #             if i < a and j < a and i != j:  # 确保是合法的边
    #                 Y[i, a+j] = 1  # 假设i到j的边映射到新矩阵的这个位置，实际应调整逻辑
    #                 Y[a+j, i] = 1  # 对称位置也置1，表示双向（虽然one-hot通常单向，但这里为了匹配描述）

    loca_list = torch.nonzero(X)
    for k in range(loca_list.shape[0]):
        i = loca_list[k][0]
        j = loca_list[k][1]
        #if i < a and j < a and i != j:  
        Y[i, a+j] = 1  
        # Y[a+j, i] = 1  
    return Y


def one_hot_to_adjacency(a, b, Y):
    """
    将形状为[a+b, a+b]的one-hot编码矩阵Y转换回形状为[a, b]的邻接矩阵X。
    
    参数:
    - a: 节点数
    - Y: 形状为[a+b, a+b]的one-hot编码矩阵
    
    返回:
    - X: 形状为[a, b]的邻接矩阵
    """
    # 初始化邻接矩阵为0
    X = torch.zeros((a, b))
    
    # 根据one-hot编码逻辑恢复邻接矩阵
    # for i in range(a):  # 遍历所有可能的起始节点
    #     for j in range(a, a+b):  # 遍历可能的目标"边"索引对应的one-hot编码位置
    #         if Y[i, j] == 1:  # 如果在one-hot编码矩阵中找到对应边
    #             # 解码one-hot信息，还原邻接关系
    #             target_node_index = j - a  # 转换索引以匹配原始节点索引
    #             X[i, target_node_index] = 1  # 设置边的存在性
    
    loca_list = torch.nonzero(Y)
    print('one_hot_to_adjacency loca_list:', loca_list.shape)
    for k in range(loca_list.shape[0]):
        i = loca_list[k][0]
        j = loca_list[k][1]
        target_node_index = j - a
        # if i >= X.shape[0] or j >= X.shape[1] or i < 0 or target_node_index < 0:
        #     print('i:{} j:{} target_node_index:{}'.format(i, j, target_node_index))
        #     import pdb
        #     pdb.set_trace()
        if i < a and j >= a:
            X[i, target_node_index] = Y[i][j]

    return X

def main(args):
    out_path = os.path.join(args.log_name ,args.dataset, datetime.now().strftime('%Y%m%d'))
    out_path = os.path.join(out_path, args.out_name)
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_path_file = os.path.join(out_path, 'output_NDCG.txt')
    if not args.debug:
        sys.stdout = open(out_path_file, 'w')
    print('out_path:', out_path, out_path_file)
    # import pdb
    # pdb.set_trace()
    print("args:", args)
    random_seed = args.random_seed
    print('random_seed:', random_seed)
    # torch.manual_seed(random_seed) # cpu
    # torch.cuda.manual_seed(random_seed) # gpu
    # np.random.seed(random_seed) # numpy
    # random.seed(random_seed) # random and transforms
    # torch.backends.cudnn.deterministic=True # cudnn
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    devi = "cuda:{}".format(args.gpu)
    print('devi:', devi)
    #devi = "cuda:1"
    device = torch.device(devi if args.cuda else "cpu")
    print('device:', device)
    a = torch.ones(100).to(device)
    print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    ### DATA LOAD ###
    train_path = args.data_path + 'train_list.npy'
    valid_path = args.data_path + 'valid_list.npy'
    test_path = args.data_path + 'test_list.npy'

    train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)
    mat = torch.FloatTensor(train_data.A)
    epps0 = mat.sum() / (mat.shape[0] * mat.shape[1])
    print('mat:', mat.shape, mat.sum(), mat.sum() / (mat.shape[0] * mat.shape[1]))
    del mat
    n_user = 3000 # sample less data
    ma = torch.FloatTensor(train_data.A)
    ma = ma[:n_user]
    ta = torch.FloatTensor(test_y_data.A)
    ta = ta[:n_user]
    print('ma:', ma.shape, len(ma), args.batch_size)
    train_dataset = data_utils.DataDiffusion(ma)
    test_dataset = data_utils.DataDiffusion(ta)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn, drop_last=True)
    test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    # train_batch_list = []
    # test_batch_list = []
    # for batch_idx, batch in enumerate(train_loader):
    #     train_batch_list.append(batch)
    # for batch_idxte, batchte in enumerate(test_loader):
    #     test_batch_list.append(batchte)
    # print('batch:{} batchte:{}'.format(train_batch_list[0].shape, test_batch_list[0].shape))
    # for i in range(len(train_batch_list)):
    #     print('i:{} abs:{}'.format(i, (abs(train_batch_list[i] - test_batch_list[i])).sum()))
    # import pdb
    # pdb.set_trace()
    OneHotMatrix = args.OneHotMatrix
    # 0: default  1: concat matrix 2: class one hot matrix
    CatOneHot = False
    if OneHotMatrix == 2:
        CatOneHot = True

    if args.tst_w_val:
        tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A))
        test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
    mask_tv = train_data + valid_y_data

    print('data ready.')


    ### Build Gaussian Diffusion ###
    if args.mean_type == 'x0':
        mean_type = gd.ModelMeanType.START_X
    elif args.mean_type == 'eps':
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % args.mean_type)



    diffusion = gd.GaussianDiffusionDiscrete(mean_type, args.noise_schedule, \
            args.noise_scale, args.noise_min, args.noise_max, args.steps, device, discrete=args.discrete, CatOneHot=CatOneHot, epps=1-epps0, args=args).to(device)
    # diffusion = gd.GaussianDiffusionAblation(mean_type, args.noise_schedule, \
    #         args.noise_scale, args.noise_min, args.noise_max, args.steps, device, discrete=args.discrete, CatOneHot=CatOneHot, epps=1-epps0, args=args).to(device)
    
    ### Build MLP ###
    out_dims = args.dims + [n_item]
    print('out1:', out_dims, args.dims, [n_item])
    # import pdb
    # pdb.set_trace()
    #out_dims = [eval(args.dims)] + [n_item]
    if OneHotMatrix == 1:
        #out_dims = [eval(args.dims)] + [n_item + args.batch_size]
        out_dims = args.dims + [n_item + args.batch_size]
    in_dims = out_dims[::-1]
    print('in_dims:{} out_dims:{}'.format(in_dims, out_dims))
    # import pdb
    # pdb.set_trace()
    #if OneHotMatrix <= 1:
    print('backbone:', args.backbone)
    if args.backbone == 'DNN':
        print('dnn')
        model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)
    elif args.backbone == 'DNNOneHot':
        model = DNNOneHot(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)
    
    elif args.backbone == 'DNNCat':
        model = DNNCat(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)

    elif args.backbone == 'lightGCN':
        diffusion.gcn = 1
        ta = torch.FloatTensor(train_data.A)
        index = torch.zeros(ta.shape[0])
        for i in range(index.shape[0]):
            index[i] = i
        index = index.long()
        e_f = adjacency_to_edge(ta, index)
        # convert_e = edge_to_adjacency(e_t, index, bs=ta.shape[0])
        # print('x_t:{} e_t:{} convert_e:{}'.format(ta.shape, e_t.shape, convert_e.shape))
        # print('equ:', (ta.cpu() == convert_e).sum()) # 16716690=5949*2810
        # print('ta:', ta.shape)
        # import pdb
        # pdb.set_trace()
        model = DNNlightGCN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm, e_f=e_f).to(device)
    
    elif args.backbone == 'DNNOneHotTransformer':
        model = DNNOneHotTransformer(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)
    
    elif args.backbone == 'DNNOneHotEmbedding':
        diffusion.indexIn = True
        model = DNNOneHotEmbedding(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm, item_num=n_item, user_num=n_user).to(device)
    elif args.backbone == 'DNNOneHotEmbeddingGCN':
        diffusion.indexIn = True
        model = DNNOneHotEmbeddingGCN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm, item_num=n_item, user_num=n_user,
                                      args=args).to(device)
    elif args.backbone == 'DNNOneHotEmbedding_conti':
        model = DNNOneHotEmbedding_conti(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm, item_num=n_item, user_num=n_user,
                                      args=args).to(device)
    elif args.backbone == 'DNNOneHotEmbeddingGCN_conti':
        diffusion.indexIn = True
        model = DNNOneHotEmbeddingGCN_conti(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm, item_num=n_item, user_num=n_user,
                                      args=args).to(device)
    else:
        print('not implemented!')
        exit()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("models ready.")

    param_num = 0
    mlp_num = sum([param.nelement() for param in model.parameters()])
    diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
    param_num = mlp_num + diff_num
    print("Number of all parameters:", param_num)

    def evaluate(data_loader, data_te, mask_his, topN):
        model.eval()
        e_idxlist = list(range(mask_his.shape[0]))
        e_N = mask_his.shape[0]

        predict_items = []
        target_items = []
        for i in range(e_N):
            target_items.append(data_te[i, :].nonzero()[1].tolist())
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                #print('batch_idx:', batch_idx)
                index = batch[1]
                batch = batch[0]
                if OneHotMatrix == 1:
                    a, b = batch.shape[0], batch.shape[1]

                    batch = adjacency_to_one_hot(batch.shape[0], batch.shape[1], batch)
                # his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
                batch = batch.to(device)
                prediction = diffusion.p_sample(model, batch, args.sampling_steps, args.sampling_noise, index=index)
                
                if OneHotMatrix == 1:
                    thres = 0.1
                    #print('prediction:', prediction.shape, prediction.min(), prediction.max(), prediction.mean())
                    #prediction[prediction > thres] = 1
                    prediction[prediction <= thres] = 0
                    prediction = one_hot_to_adjacency(a, b, prediction)
                his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(prediction)]]
                # print('hid_data:{} pred:{}'.format(his_data.shape, prediction.shape))
                
                prediction[his_data.nonzero()] = -np.inf

                _, indices = torch.topk(prediction, topN[-1])
                # import pdb
                # pdb.set_trace()
                indices = indices.cpu().numpy().tolist()
                predict_items.extend(indices)

        test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)
        # import pdb
        # pdb.set_trace()
        return test_results

    best_recall, best_epoch = -100, 0
    best_test_result = None
    print("Start training...")
    for epoch in range(1, args.epochs + 1):
        if epoch - best_epoch >= 200:
            print('-'*18)
            print('Exiting from training early')
            break

        model.train()
        start_time = time.time()

        batch_count = 0
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            continue
            # import pdb
            # pdb.set_trace()
            index = batch[1]
            batch = batch[0]
            #print('batch:', batch.shape)
            if OneHotMatrix == 1:
                a, b = batch.shape[0], batch.shape[1]
                batch = adjacency_to_one_hot(a, b, batch)
                # batch2 = batch
                #batch3 = one_hot_to_adjacency(a, b, batch2)
                # print('batch-2:', (batch == batch3).sum())
                # import pdb
                # pdb.set_trace()
            #print('batch2:', batch.shape)
            batch = batch.to(device)
            batch_count += 1
            optimizer.zero_grad()
            losses = diffusion.training_losses(model, batch, args.reweight, index=index)
            #print('losses:', losses)
            loss = losses["loss"].mean()
            total_loss += loss
            loss.backward()
            optimizer.step()
        
        if epoch % 5 == 0:
            valid_results = evaluate(test_loader, valid_y_data, train_data, eval(args.topN))
            if args.tst_w_val:
                test_results = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN))
            else:
                test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
            evaluate_utils.print_results(None, valid_results, test_results)
            sys.stdout.flush()

            if valid_results[2][1] > best_recall: # recall@20 as selection
                best_recall, best_epoch = test_results[2][1], epoch
                best_results = test_results
                best_test_results = test_results

                # if not os.path.exists(args.save_path):
                #     os.makedirs(args.save_path)
                # torch.save(model, '{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_{}.pth' \
                #     .format(args.save_path, args.dataset, args.lr, args.weight_decay, args.batch_size, args.dims, args.emb_size, args.mean_type, \
                #     args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.log_name))
                
                model_path = os.path.join(out_path, 'model.pth')
                print('model_path:', model_path)
                torch.save(model, model_path)
        
        print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                            "%H: %M: %S", time.gmtime(time.time()-start_time)))
        print('---'*18)

    print('==='*18)
    print("End. Best Epoch {:03d} ".format(best_epoch))
    evaluate_utils.print_results(None, None, best_test_results)   
    print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))





if __name__ == '__main__':
    _args = parse_args()
    #print('args:', _args)
    main(_args)
