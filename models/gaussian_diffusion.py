import enum
import math
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import random, time
from data_utils import adjacency_to_edge, edge_to_adjacency, topk_set, set_top_k_to_one

class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

class AbsorbingStateTransition:
    def __init__(self, num_nodes=400):
        self.num_nodes = num_nodes
        self.u_x = th.ones(1, self.num_nodes, self.num_nodes)

    def get_Qt_bar(self, alpha_bar_t):
        """ beta_t: (bs)
        Returns transition matrices for X and E"""

        alpha_bar_t = alpha_bar_t.unsqueeze(1)

        q_x = alpha_bar_t * th.eye(self.X_classes).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x

        return q_x





def mix_tensors(tensor1, tensor2, mix_prob=0.5):
    """
    按照给定的概率mix_prob混合两个张量的元素。

    参数:
    tensor1, tensor2 (torch.Tensor): 要混合的两个张量，要求形状相同。
    mix_prob (float): 混合每个元素的概率，默认为0.5。

    返回:
    torch.Tensor: 混合后的张量。
    """
    assert tensor1.shape == tensor2.shape, "两个张量的形状必须一致"
    
    # 生成与tensor1形状相同的随机掩码，其中元素为0或1
    mask = th.zeros_like(tensor1).bernoulli(mix_prob)
    
    # 使用掩码混合两个张量
    mixed_tensor = mask * tensor1 + (1 - mask) * tensor2
    
    return mixed_tensor

class GaussianDiffusion(nn.Module):
    def __init__(self, mean_type, noise_schedule, noise_scale, noise_min, noise_max,\
            steps, device, history_num_per_term=10, beta_fixed=True, discrete=0.99, CatOneHot=False):

        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device
        self.discrete = discrete
        self.CatOneHot = CatOneHot
        self.gcn = 0

        self.history_num_per_term = history_num_per_term
        self.Lt_history = th.zeros(steps, history_num_per_term, dtype=th.float64).to(device)
        self.Lt_count = th.zeros(steps, dtype=int).to(device)
        self.num_nodes = 2810
        
        #self.TransMatrix = AbsorbingStateTransition().to(device)

        if noise_scale != 0.: 
            self.betas = th.tensor(self.get_betas(), dtype=th.float64).to(self.device)# controls the noise scales added at the step t
            if beta_fixed:
                self.betas[0] = 0.00001  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
                # The variance \beta_1 of the first step is fixed to a small constant to prevent overfitting.
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"

            self.calculate_for_diffusion()

        super(GaussianDiffusion, self).__init__()
        #self.u_x = nn.Parameter(th.ones(1, self.num_nodes, self.num_nodes))
        self.u_x = th.ones(1, 2, 2)
    
    def get_Qt_bar(self, alpha_bar_t):
        """ beta_t: (bs)
        Returns transition matrices for X and E"""

        #alpha_bar_t = alpha_bar_t.unsqueeze(1)
        # print('alpha_bar_t:', alpha_bar_t.shape)
        # print('self.u_x:', self.u_x.shape)
        # print('eye:', th.eye(self.num_nodes).unsqueeze(0).shape)

        alpha_bar_t = alpha_bar_t.unsqueeze(-1).unsqueeze(-1).expand(-1, self.u_x.shape[1], self.u_x.shape[1])
        
        # print('alpha_bar_t_ex:', alpha_bar_t.shape, alpha_bar_t.min(), alpha_bar_t.max())
        # import pdb
        # pdb.set_trace()
        q_x = alpha_bar_t * th.eye(self.num_nodes).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x

        return q_x

    def get_betas(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        elif self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
            self.steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        elif self.noise_schedule == "binomial":  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")
    
    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = th.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = th.cat([th.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)  # alpha_{t-1}
        self.alphas_cumprod_next = th.cat([self.alphas_cumprod[1:], th.tensor([0.0]).to(self.device)]).to(self.device)  # alpha_{t+1}
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = th.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = th.log(
            th.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * th.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * th.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    def p_sample(self, model, x_start, steps, sampling_noise=False, index=None):
        assert steps <= self.steps, "Too much steps in inference."
        #import time
        #t0 = time.time()
        if self.CatOneHot:
            x_start_one = F.one_hot(x_start.long(), num_classes=2)
            x_start_one = x_start_one.float()
            #x_startU = x_start.unsqueeze(-1)
            #print('x_one:{} x_U:{}'.format(x_start_one.shape, x_startU.shape))
            # x_startU = th.cat([x_startU, x_start_one], dim=2)
            if steps == 0:
                x_tU = x_start_one
            else:
                t = th.tensor([steps - 1] * x_start_one.shape[0]).to(x_start_one.device)
                x_tU = self.q_sample(x_start_one, t)
        if steps == 0:
            x_t = x_start
        else:
            t = th.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)
            # x_t = self.apply_noise(t, x_start, x_base=x_t)
            # x_t = th.where(x_t > self.discrete, 1.0, 0)
        #print('x_t:', x_t.sum(), steps)
        #t1 = time.time()
        #print('time noise:', t1- t0)
        indices = list(range(self.steps))[::-1]
        #print('indices:{} self.steps:{}'.format(indices, self.steps))
        if self.noise_scale == 0.:
            for i in indices:
                t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
                if self.gcn:
                    x_t = model(x_t, t, index)
                else:
                    x_t = model(x_t, t, x_tU)
            return x_t
        #t2 = time.time()
        #print('time model:', t2-t1)
        for i in indices:
            #print('i:', i)
            #print('i:{} sampling_noise:{}'.format(i, sampling_noise))
            
            # x_t = set_top_k_to_one(x_t)
            t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
            if not self.CatOneHot:
                x_tU = None
            if self.gcn:
                out = self.p_mean_variance(model, x_t, t, index=index)
            else:
                out = self.p_mean_variance(model, x_t, t, x_tU=x_tU)
            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        # t3 = time.time()
        # print('time sample:', t3-t2)
        return x_t
    
    def apply_noise(self, ts, x_start, x_base=None):
        # import time
        # time_s = time.time()

        batch_size, device = x_start.size(0), x_start.device
        tsF = ts.float() / batch_size
        Qtb = self.get_Qt_bar(tsF)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        # time1 = time.time()
        # print('time get Q:', time1 - time_s)
        Qtb = Qtb.to(device)
        # time2 = time.time()
        # print('Qt to', time2 - time1)
        # print('Qtb:{} x_start:{}'.format(Qtb.shape, x_start.shape))
        # print('Qtb:', Qtb.shape, Qtb.min(), Qtb.max(), Qtb.mean())
        # Compute transition probabilities
        #probX = x_start @ Qtb  # (bs, n, dx_out)
        probX = x_start.unsqueeze(-1) * Qtb
        # time3 = time.time()
        # print('probX time:', time3 - time2)
        # print('probX:{}'.format(probX.shape))
        # print('probX:', probX.shape, probX.min(), probX.max(), probX.mean())
        # print('probX0:', probX[0].sum())
        # print('probX00:', probX[0,0].sum())
        #print('probX:{} probX2:{}'.format(probX.shape, probX2.shape))
        # import pdb
        # pdb.set_trace()
        

        sampled_t = self.sample_discrete_features(probX)

        # time4 = time.time()
        # print('sample time:', time4 - time3)
        # print('sampled_t:', sampled_t.shape)
        discrete = random.randint(int(sampled_t.shape[1] * 0.8), sampled_t.shape[1])
        x_t = th.where(sampled_t > discrete, 1.0, 0)
        # print('x_t:', x_t.shape)
        # import pdb
        # pdb.set_trace()
        #x_t = F.one_hot(sampled_t, num_classes=sampled_t.shape[2])

        # x_t = th.where(x_t > self.discrete, 1.0, 0)
        # result = torch.where(x > threshold, torch.tensor(1.), torch.tensor(0.))
        # print('noise:', noise.shape, noise.max(), noise.min(), noise.mean())
        # print('x_t:', x_t.shape, x_t.max(), x_t.min(), x_t.mean())
        # print('x_start:', x_start.shape, x_start.max(), x_start.min(), x_start.mean())
        # import pdb
        # pdb.set_trace()
        if x_base is None:
            x_t = mix_tensors(x_start, x_t, 0.8)
        else:
            x_t = mix_tensors(x_base, x_t, 0.99)
        return x_t
        
    
    def training_losses(self, model, x_start, reweight=False, index=None):
        # import time
        # time_s = time.time()

        if self.CatOneHot:
            #print('x_start:', x_start.shape)
            x_startU = F.one_hot(x_start.long(), num_classes=2)
            x_startU = x_startU.float()
            batch_size, device = x_startU.size(0), x_startU.device
            ts, pt = self.sample_timesteps(batch_size, device, 'importance')
            # time1 = time.time()
            # print('time sample:', time1 - time_s)
            noise = th.randn_like(x_startU)
            if self.noise_scale != 0.:
                x_tU = self.q_sample(x_startU, ts, noise)
            else:
                x_tU = x_startU
            
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        # time1 = time.time()
        # print('time sample:', time1 - time_s)
        noise = th.randn_like(x_start)
        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start
        #x_t = th.where(x_t > self.discrete, 1.0, 0)
        #print('x_start:', x_start.shape, x_start.sum())
        # x_t = set_top_k_to_one(x_t, k=25000) # random.randint(10000, 25000)
        # a, b = x_t.shape
        # e_t = adjacency_to_edge(x_t, index)
        # convert_e = edge_to_adjacency(e_t, index)
        # print('x_t:{} e_t:{} convert_e:{}'.format(x_t.shape, e_t.shape, convert_e.shape))
        # print('equ:', (x_t.cpu() == convert_e).sum())
        # import pdb
        # pdb.set_trace()
        # print('ts:', tsF.shape)
        # time2 = time.time()
        # print('time add noise:', time2 - time1)
        # x_t = self.apply_noise(ts, x_start, x_base=x_t)
        # time3 = time.time()
        # print('time discrete noise:', time3 - time2)

        
        terms = {}
        if self.CatOneHot:
            model_output = model(x_t, ts, x_tU)
        elif self.gcn:
            model_output = model(x_t, ts, index)
        else:
            model_output = model(x_t, ts)
        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        mse = mean_flat((target - model_output) ** 2)
        # print('self.alphas_cumprod:', self.alphas_cumprod.shape)
        # import pdb
        # pdb.set_trace()
        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = th.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / ((1-self.alphas_cumprod_prev[ts])**2 * (1-self.betas[ts]))
                weight = th.where((ts == 0), 1.0, weight)
                likelihood = mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output))**2 / 2.0)
                loss = th.where((ts == 0), likelihood, mse)
        else:
            weight = th.tensor([1.0] * len(target)).to(device)

        terms["loss"] = weight * loss
        
        # update Lt_history & Lt_count
        for t, loss in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss)
                    raise ValueError

        terms["loss"] /= pt
        return terms

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':  # importance sampling
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method='uniform')
            
            Lt_sqrt = th.sqrt(th.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / th.sum(Lt_sqrt)
            pt_all *= 1- uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1. < 1e-5

            t = th.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt
        
        elif method == 'uniform':  # uniform sampling
            t = th.randint(0, self.steps, (batch_size,), device=device).long()
            pt = th.ones_like(t).float()

            return t, pt
            
        else:
            raise ValueError
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    

    def sample_discrete_features(self, probX):
        ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
            :param probX: bs, n, dx_out        node features
            :param probE: bs, n, n, de_out     edge features
            :param proby: bs, dy_out           global features.
        '''
        # import time
        # times = time.time()
        # print('sample begin:')
        bs, n, _ = probX.shape
        # print('probX:', probX.shape)
        # print('probX0', probX[0].sum())
        # Noise X
        # The masked rows should define probability distributions as well
        # probX[~node_mask] = 1 / probX.shape[-1]

        # Flatten the probability tensor to sample with multinomial
        probX = probX.reshape(bs * n, -1) + 1e-5   # (bs * n, dx_out)
        #print('probX0', probX[0].sum())
        sums_per_row = probX.sum(dim=1, keepdim=True) + 1e-5
        
        
        # 确保求和不为零（避免除以零错误）
        # sums_per_row = th.where(sums_per_row == 0, th.ones_like(sums_per_row), sums_per_row)

        # 归一化，使每行之和为1
        probX = probX / sums_per_row 
        # time1 = time.time()
        # print('time norm:', time1 - times)
        #print('probX:', probX.shape, probX.min(), probX.max(), probX.mean())
        # import pdb
        # pdb.set_trace()
        # Sample X
        X_t = probX.multinomial(1)    # (bs * n, 1)
        # time2 = time.time()                              
        # print('time mul:', time2 - time1)
        X_t = X_t.reshape(bs, n)     # (bs, n)
        # import pdb
        # pdb.set_trace()
        return X_t
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x, t, x_tU=None,index=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x.shape[:2]
        #print('x:', x.shape, x.min(), x.max(), x.mean(), x.sum())
        
        assert t.shape == (B, )
        if self.CatOneHot:
            model_output = model(x, t, x_tU)
        elif self.gcn:
            model_output = model(x, t, index)
        else:
            model_output = model(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        
        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)
        #print('mean_type:', self.mean_type)
        #print('model_output:', model_output.shape, model_output.min(), model_output.max(), model_output.mean(), model_output.sum())
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        #print('model_mean:', model_mean.shape, model_mean.min(), model_mean.max(), model_mean.mean(), model_mean.sum())

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)




class GaussianDiffusionDiscrete(nn.Module):
    def __init__(self, mean_type, noise_schedule, noise_scale, noise_min, noise_max,\
            steps, device, history_num_per_term=10, beta_fixed=True, discrete=0.99, CatOneHot=False, epps = 0.9995, args=None):
        self.args = args
        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device
        self.discrete = discrete
        self.discrete_noise = True
        self.CatOneHot = CatOneHot

        self.history_num_per_term = history_num_per_term
        self.Lt_history = th.zeros(steps, history_num_per_term, dtype=th.float64).to(device)
        self.Lt_count = th.zeros(steps, dtype=int).to(device)
        self.num_nodes = 2810
        self.indexIn = False
        
        #self.TransMatrix = AbsorbingStateTransition().to(device)

        if noise_scale != 0.: 
            self.betas = th.tensor(self.get_betas(), dtype=th.float64).to(self.device)# controls the noise scales added at the step t
            if beta_fixed:
                self.betas[0] = 0.00001  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
                # The variance \beta_1 of the first step is fixed to a small constant to prevent overfitting.
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"

            self.calculate_for_diffusion()

        super(GaussianDiffusionDiscrete, self).__init__()
        #self.u_x = nn.Parameter(th.ones(1, self.num_nodes, self.num_nodes))
        #self.u_x = th.ones(1, 2, 2)
        epps = self.discrete
        print('epps:', epps)
        self.u_x = th.tensor([[epps, 1 - epps],
                              [epps, 1 - epps]])
        self.u_x = self.u_x.unsqueeze(0).to(device)
        self.u_x_eye = th.eye(self.u_x.shape[1]).unsqueeze(0).to(device)
        self.criterion = nn.BCELoss()
    
    def get_Qt_bar(self, alpha_bar_t):
        """ beta_t: (bs)
        Returns transition matrices for X and E"""

        alpha_bar_t = alpha_bar_t.unsqueeze(1).unsqueeze(1)
        # print('alpha_bar_t:', alpha_bar_t.shape)
        #print('self.u_x:', self.u_x.shape, self.u_x.device)
        # print('eye:', th.eye(self.num_nodes).unsqueeze(0).shape)

        # alpha_bar_t = alpha_bar_t.unsqueeze(-1).unsqueeze(-1).expand(-1, self.u_x.shape[1], self.u_x.shape[1])
        
        # print('alpha_bar_t_ex:', alpha_bar_t.shape, alpha_bar_t.min(), alpha_bar_t.max())
        # import pdb
        # pdb.set_trace()
        #print('alpha_bar_t:', alpha_bar_t.shape, alpha_bar_t.device)
        q_x = alpha_bar_t * self.u_x_eye + (1 - alpha_bar_t) * self.u_x

        return q_x

    def get_betas(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        elif self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
            self.steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        elif self.noise_schedule == "binomial":  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")
    
    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = th.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = th.cat([th.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)  # alpha_{t-1}
        self.alphas_cumprod_next = th.cat([self.alphas_cumprod[1:], th.tensor([0.0]).to(self.device)]).to(self.device)  # alpha_{t+1}
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = th.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = th.log(
            th.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * th.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * th.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    def p_sample(self, model, x_start, steps, sampling_noise=False,index=None):
        assert steps <= self.steps, "Too much steps in inference."
        time_start = time.time()
        if self.CatOneHot:
            x_start_one = F.one_hot(x_start.long(), num_classes=2)
            x_start_one = x_start_one.float()
            # x_startU = x_start.unsqueeze(-1)
            # print('x_one:{} x_U:{}'.format(x_start_one.shape, x_startU.shape))
            # x_startU = th.cat([x_startU, x_start_one], dim=2)
            # print('steps:', steps)
            if steps == 0:
                x_tU = x_start_one
            else:
                t = th.tensor([steps - 1] * x_start_one.shape[0]).to(x_start_one.device)
                if self.discrete_noise:
                    x_tU = self.apply_noise(t, x_start_one)
                    x_tU = x_tU & (F.one_hot(x_start.long(), num_classes=2))

                else:
                    x_tU = self.q_sample(x_start_one, t)
        if steps == 0:
            x_t = x_start
        else:
            t = th.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)
            # x_t = self.apply_noise(t, x_start, x_base=x_t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = model(x_t, t, x_tU)
            return x_t
        x_start_zero = F.one_hot(th.zeros_like(x_start.long()), num_classes=2)
        x_start_zero = x_start_zero.float()
        time_noise = time.time()
        #print('time noise:', time_noise - time_start)
        for i in indices:
            t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
            if not self.CatOneHot:
                x_tU = None
            x_start_i = self.apply_noise(t, x_start_zero.float())
            x_degree = x_start.sum(dim=1)
            x_degree = x_degree / (x_degree.max())
            x_degree = x_degree.unsqueeze(1)
            x_degree_sub = 1 - x_degree
            x_degree_mat = th.cat([x_degree_sub, x_degree], dim=1)
            x_degree_sample = x_degree_mat.multinomial(1)
            x_degree_sample = x_degree_sample.repeat_interleave(x_start_i.shape[1], dim=1)
            x_degree_sample_hot = F.one_hot(x_degree_sample, num_classes=2)
            #print('x_degree_sample_hot:', x_degree_sample_hot.shape)
            if self.args.user_guided:
                x_start_io = (x_start_i & x_degree_sample_hot)
            else:
                x_start_io = x_start_i
            x_start_zerog = x_start_zero.argmax(dim=2)
            x_start_iog = x_start_io.argmax(dim=2)
            x_start_ion = (x_start_iog.long() | x_start_zerog.long())
            x_start_ionh = F.one_hot(x_start_ion, num_classes=2)

            x_start_zero = x_start_ionh
            time_graph = time.time()
            #print('{} time graph:{}'.format(i, time_graph - time_noise))
            # print('x_start:{} x_degree:{} x_start_zero:{} x_start_ionh:{}'.format(x_start_i[:,:,1].sum(), x_degree_sample_hot[:,:,1].sum(),
            #                                                                       x_start_zero[:,:,1].sum(), x_start_ionh[:, :, 1].sum()))
            # print('sample:', x_degree_sample.shape, x_degree_sample.sum())
            # print('x_degree:', x_degree.shape, x_degree.max(), x_degree.min(), x_degree.mean())
            # x_tU = x_tU & (F.one_hot(x_start.long(), num_classes=2))
            
            # print('x_start_one:{}', x_start_one.shape, x_start_one.sum())
            # print('x_start_zero:{}', x_start_zero.shape, x_start_zero.sum())
            # import pdb
            # pdb.set_trace()

            #print('graph:', i, x_start_ion.shape, x_start_ion.sum())
            out = self.p_mean_variance(model, x_t, t, x_tU=x_tU, index=index.cuda(), graph=x_start_ionh)
            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
            time_model = time.time()
            #print('time modelT:', time_model - time_graph)
            # import pdb
            # pdb.set_trace()
            # from thop import profile
            # from thop import clever_format
            # macs, params = profile(model, inputs=(x_t, t, x_tU, index.cuda(), x_start_ionh))  # macs 表示乘法加法操作数
            # macs, params = clever_format([macs, params], "%.3f")
            #print('i:', i)
            # print(f"Computational complexity: {macs} M")
            # print(f"Number of parameters: {params} M")
        time_infer = time.time()
        #print('time_infer:', time_infer -time_noise)
        # import pdb
        # pdb.set_trace()
        return x_t
    
    def apply_noise(self, ts, x_start, x_base=None):
        # import time
        # time_s = time.time()

        batch_size, device = x_start.size(0), x_start.device
        tsF = ts.float() / batch_size
        Qtb = self.get_Qt_bar(tsF)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        # time1 = time.time()
        # print('time get Q:', time1 - time_s)
        Qtb = Qtb.to(device)
        # time2 = time.time()
        # print('Qt to', time2 - time1)
        # print('Qtb:{} x_start:{}'.format(Qtb.shape, x_start.shape))
        #print('Qtb:', Qtb.shape, Qtb.min(), Qtb.max(), Qtb.mean())
        # Compute transition probabilities
        # probX = x_start @ Qtb  # (bs, n, dx_out)
        #print('x_start:', x_start.shape)
        # import pdb
        # pdb.set_trace()
        # probX = x_start.unsqueeze(-1) * Qtb
        probX = x_start @ Qtb
        # time3 = time.time()
        # print('probX time:', time3 - time2)
        #print('probX:{}'.format(probX.shape))
        # print('probX:', probX.shape, probX.min(), probX.max(), probX.mean())
        # print('probX0:', probX[0].sum())
        # print('probX00:', probX[0,0].sum())
        # print('probX:{} probX2:{}'.format(probX.shape, probX2.shape))
        # import pdb
        # pdb.set_trace()
        

        sampled_t = self.sample_discrete_features(probX)

        # time4 = time.time()
        # print('sample time:', time4 - time3)
        #print('sampled_t:', sampled_t.shape)
        # discrete = random.randint(int(sampled_t.shape[1] * 0.8), sampled_t.shape[1])
        # x_t = th.where(sampled_t > discrete, 1.0, 0)
        x_t = F.one_hot(sampled_t, num_classes=2)

        # ct = x_t.argmax(dim=2)
        # print('x_t:{} ct:{} sampled_t:{}'.format(x_t.shape, ct.shape, sampled_t.shape))
        # print('x_start:', x_start.sum(), x_start.shape)
        # print('sampled_t:', sampled_t.sum(), sampled_t.shape)
        # dt = x_start.argmax(dim=2)
        # print('noise sum:', (dt==ct).sum())
        # import pdb
        # pdb.set_trace()

        # x_t = th.where(x_t > self.discrete, 1.0, 0)
        # result = torch.where(x > threshold, torch.tensor(1.), torch.tensor(0.))
        # print('noise:', noise.shape, noise.max(), noise.min(), noise.mean())
        #print('x_t:', x_t.shape, x_t.max(), x_t.min())
        # print('x_start:', x_start.shape, x_start.max(), x_start.min(), x_start.mean())
        # import pdb
        # pdb.set_trace()
        # if x_base is None:
        #     x_t = mix_tensors(x_start, x_t, 0.8)
        # else:
        #     x_t = mix_tensors(x_base, x_t, 0.99)
        return x_t
        
    
    def training_losses(self, model, x_start, reweight=False, index=None):
        #print('train batch in')
        # import time
        # time_s = time.time()

        if self.CatOneHot:
            #print('x_start:', x_start.shape)
            #print('x_startN:', x_start.sum(), x_start.shape)
            x_startU = F.one_hot(x_start.long(), num_classes=2)
            x_startU = x_startU.float()
            batch_size, device = x_startU.size(0), x_startU.device
            ts, pt = self.sample_timesteps(batch_size, device, 'importance')
            # time1 = time.time()
            # print('time sample:', time1 - time_s)
            if self.discrete_noise:
                x_tU = self.apply_noise(ts, x_startU)
                
                x_tU = x_tU & (F.one_hot(x_start.long(), num_classes=2))
                x_tU = x_tU.float()
                # print('x_tU:{} ts:{} x_startU:{}'.format(x_tU.shape, ts.shape, x_startU.shape))
                # print('equal per:', ((x_tU == x_startU).sum()) / (x_startU.shape[0] * x_startU.shape[1] * x_startU.shape[2]))
                # import pdb
                # pdb.set_trace()
            else:
                noise = th.randn_like(x_startU)
                if self.noise_scale != 0.:
                    x_tU = self.q_sample(x_startU, ts, noise)
                else:
                    x_tU = x_startU
            
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        # time1 = time.time()
        # print('time sample:', time1 - time_s)
        noise = th.randn_like(x_start)
        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        # print('eps per:', ((x_t - x_start < 0.01).sum()) / (x_start.shape[0] * x_start.shape[1]))
        # print('x_start:', x_start.shape)
        # print('ts:', tsF.shape)
        # time2 = time.time()
        # print('time add noise:', time2 - time1)
        # x_t = self.apply_noise(ts, x_start, x_base=x_t)
        # time3 = time.time()
        # print('time discrete noise:', time3 - time2)

        Closs = None
        terms = {}
        if self.CatOneHot:
            if self.indexIn:
                #model_output = model(x_t, ts, x_tU, index=index, graph=x_startU)
                # model_output, Closs = model(x_t, ts, x_tU, index=index, graph=x_startU, RCloss=True)
                model_output, Closs = model(x_t, ts, x_tU, index=index.cuda(), graph=x_tU.long(), RCloss=True)
            else:
                model_output = model(x_t, ts, x_tU)
                #Closs = 0
        else:
            model_output = model(x_t, ts)
        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        mse = mean_flat((target - model_output) ** 2)
        #print('target:{} output:{} mse:{}'.format(target.shape, model_output.shape, mse))
        # if self.CatOneHot:
        #     print('x_t:{} x_tU:{}'.format(x_t.shape, x_tU.shape))
        #     x_tUd = x_tU.argmax(dim=2)
        #     print('x_tUd:', x_tUd.shape, (x_tUd == x_start.long()).sum())
            
        #     Closs = nt_xent_loss(x_t, x_tUd.float())
        #     print('Closs:{} mse:{}'.format(Closs, mse))
            
        #outputs = torch.sigmoid(model(inputs))
        #bce = self.criterion(th.sigmoid(model_output), target)
        # print('mse:{} bce:{}'.format(mse.mean(), bce.mean()))
        # import pdb
        # pdb.set_trace()
        #mse = bce * 0.05

        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = th.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / ((1-self.alphas_cumprod_prev[ts])**2 * (1-self.betas[ts]))
                weight = th.where((ts == 0), 1.0, weight)
                likelihood = mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output))**2 / 2.0)
                loss = th.where((ts == 0), likelihood, mse)
        else:
            weight = th.tensor([1.0] * len(target)).to(device)
        #print('weight:{} loss:{}'.format(weight, loss))
        terms["loss"] = weight * loss
        
        # update Lt_history & Lt_count
        for t, loss in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                #print('t:', loss.detach())
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                    self.Lt_count[t] += 1
                except:
                    #print(t)
                    #print(self.Lt_count[t])
                    #print(loss)
                    raise ValueError

        terms["loss"] /= pt
        if Closs is not None:
            terms["loss"] += Closs * 0.1
        # print('loss:{} Closs:{}'.format(terms['loss'].mean(), Closs))
        # import pdb
        # pdb.set_trace()
        return terms

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':  # importance sampling
            if not (self.Lt_count == self.history_num_per_term).all():
                #print('in:')
                return self.sample_timesteps(batch_size, device, method='uniform')
            
            Lt_sqrt = th.sqrt(th.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / th.sum(Lt_sqrt)
            pt_all *= 1- uniform_prob
            pt_all += uniform_prob / len(pt_all)
            #print('pt_all:', pt_all.shape, pt_all.sum(-1), self.Lt_history.sum())
            # import pdb
            # pdb.set_trace()
            assert pt_all.sum(-1) - 1. < 1e-5

            t = th.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt
        
        elif method == 'uniform':  # uniform sampling
            t = th.randint(0, self.steps, (batch_size,), device=device).long()
            pt = th.ones_like(t).float()

            return t, pt
            
        else:
            raise ValueError
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    

    def sample_discrete_features(self, probX):
        ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
            :param probX: bs, n, dx_out        node features
            :param probE: bs, n, n, de_out     edge features
            :param proby: bs, dy_out           global features.
        '''
        # import time
        # times = time.time()
        # print('sample begin:')
        bs, n, _ = probX.shape
        # print('probX:', probX.shape)
        # print('probX0', probX[0].sum())
        # Noise X
        # The masked rows should define probability distributions as well
        # probX[~node_mask] = 1 / probX.shape[-1]

        # Flatten the probability tensor to sample with multinomial
        probX = probX.reshape(bs * n, -1) #+ 1e-5   # (bs * n, dx_out)
        #print('probX0', probX[0].sum())
        #sums_per_row = probX.sum(dim=1, keepdim=True) + 1e-5
        
        
        # 确保求和不为零（避免除以零错误）
        # sums_per_row = th.where(sums_per_row == 0, th.ones_like(sums_per_row), sums_per_row)

        # 归一化，使每行之和为1
        #probX = probX / sums_per_row 
        # time1 = time.time()
        # print('time norm:', time1 - times)
        #print('probX:', probX.shape, probX.min(), probX.max(), probX.mean())
        # import pdb
        # pdb.set_trace()
        # Sample X
        X_t = probX.multinomial(1)    # (bs * n, 1)
        #print('x_t:', X_t.shape)
        # time2 = time.time()                              
        # print('time mul:', time2 - time1)
        X_t = X_t.reshape(bs, n)     # (bs, n)
        # import pdb
        # pdb.set_trace()
        return X_t
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x, t, x_tU=None, index=None, graph=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B, )
        
        if self.CatOneHot:
            if self.indexIn:
                model_output = model(x, t, x_tU, index=index, graph=graph)
            else:
                model_output = model(x, t, x_tU)
        else:
            model_output = model(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        
        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)
        
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class GaussianDiffusionAblation(nn.Module):
    def __init__(self, mean_type, noise_schedule, noise_scale, noise_min, noise_max,\
            steps, device, history_num_per_term=10, beta_fixed=True, discrete=0.99, CatOneHot=False, epps = 0.9995, args=None):
        self.args = args
        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device
        self.discrete = discrete
        self.discrete_noise = True
        self.CatOneHot = CatOneHot

        self.history_num_per_term = history_num_per_term
        self.Lt_history = th.zeros(steps, history_num_per_term, dtype=th.float64).to(device)
        self.Lt_count = th.zeros(steps, dtype=int).to(device)
        self.num_nodes = 2810
        self.indexIn = False
        
        #self.TransMatrix = AbsorbingStateTransition().to(device)

        if noise_scale != 0.: 
            self.betas = th.tensor(self.get_betas(), dtype=th.float64).to(self.device)# controls the noise scales added at the step t
            if beta_fixed:
                self.betas[0] = 0.00001  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
                # The variance \beta_1 of the first step is fixed to a small constant to prevent overfitting.
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"

            self.calculate_for_diffusion()

        super(GaussianDiffusionAblation, self).__init__()
        #self.u_x = nn.Parameter(th.ones(1, self.num_nodes, self.num_nodes))
        #self.u_x = th.ones(1, 2, 2)
        epps = self.discrete
        print('epps:', epps)
        self.u_x = th.tensor([[epps, 1 - epps],
                              [epps, 1 - epps]])
        self.u_x = self.u_x.unsqueeze(0).to(device)
        self.u_x_eye = th.eye(self.u_x.shape[1]).unsqueeze(0).to(device)
        self.criterion = nn.BCELoss()
    
    def get_Qt_bar(self, alpha_bar_t):
        """ beta_t: (bs)
        Returns transition matrices for X and E"""

        alpha_bar_t = alpha_bar_t.unsqueeze(1).unsqueeze(1)
        # print('alpha_bar_t:', alpha_bar_t.shape)
        #print('self.u_x:', self.u_x.shape, self.u_x.device)
        # print('eye:', th.eye(self.num_nodes).unsqueeze(0).shape)

        # alpha_bar_t = alpha_bar_t.unsqueeze(-1).unsqueeze(-1).expand(-1, self.u_x.shape[1], self.u_x.shape[1])
        
        # print('alpha_bar_t_ex:', alpha_bar_t.shape, alpha_bar_t.min(), alpha_bar_t.max())
        # import pdb
        # pdb.set_trace()
        #print('alpha_bar_t:', alpha_bar_t.shape, alpha_bar_t.device)
        q_x = alpha_bar_t * self.u_x_eye + (1 - alpha_bar_t) * self.u_x

        return q_x

    def get_betas(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        elif self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
            self.steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        elif self.noise_schedule == "binomial":  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")
    
    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = th.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = th.cat([th.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)  # alpha_{t-1}
        self.alphas_cumprod_next = th.cat([self.alphas_cumprod[1:], th.tensor([0.0]).to(self.device)]).to(self.device)  # alpha_{t+1}
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = th.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = th.log(
            th.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * th.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * th.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    def p_sample(self, model, x_start, steps, sampling_noise=False,index=None):
        assert steps <= self.steps, "Too much steps in inference."

        if self.CatOneHot:
            x_start_one = F.one_hot(x_start.long(), num_classes=2)
            x_start_one = x_start_one.float()
            # x_startU = x_start.unsqueeze(-1)
            # print('x_one:{} x_U:{}'.format(x_start_one.shape, x_startU.shape))
            # x_startU = th.cat([x_startU, x_start_one], dim=2)
            # print('steps:', steps)
            if steps == 0:
                x_tU = x_start_one
            else:
                t = th.tensor([steps - 1] * x_start_one.shape[0]).to(x_start_one.device)
                if self.discrete_noise:
                    x_tU = self.apply_noise(t, x_start_one)
                    x_tU = x_tU & (F.one_hot(x_start.long(), num_classes=2))

                else:
                    x_tU = self.q_sample(x_start_one, t)
        if steps == 0:
            x_t = x_start
        else:
            t = th.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)
            # x_t = self.apply_noise(t, x_start, x_base=x_t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = model(x_t, t, x_tU)
            return x_t
        x_start_zero = F.one_hot(th.zeros_like(x_start.long()), num_classes=2)
        x_start_zero = x_start_zero.float()
        for i in indices:
            t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
            if not self.CatOneHot:
                x_tU = None
            x_start_i = self.apply_noise(t, x_start_zero.float())
            x_degree = x_start.sum(dim=1)
            x_degree = x_degree / (x_degree.max())
            x_degree = x_degree.unsqueeze(1)
            x_degree_sub = 1 - x_degree
            x_degree_mat = th.cat([x_degree_sub, x_degree], dim=1)
            x_degree_sample = x_degree_mat.multinomial(1)
            x_degree_sample = x_degree_sample.repeat_interleave(x_start_i.shape[1], dim=1)
            x_degree_sample_hot = F.one_hot(x_degree_sample, num_classes=2)
            #print('x_degree_sample_hot:', x_degree_sample_hot.shape)
            x_start_io = (x_start_i & x_degree_sample_hot)
            x_start_zerog = x_start_zero.argmax(dim=2)
            x_start_iog = x_start_io.argmax(dim=2)
            x_start_ion = (x_start_iog.long() | x_start_zerog.long())
            x_start_ionh = F.one_hot(x_start_ion, num_classes=2)

            x_start_zero = x_start_ionh
            # print('x_start:{} x_degree:{} x_start_zero:{} x_start_ionh:{}'.format(x_start_i[:,:,1].sum(), x_degree_sample_hot[:,:,1].sum(),
            #                                                                       x_start_zero[:,:,1].sum(), x_start_ionh[:, :, 1].sum()))
            # print('sample:', x_degree_sample.shape, x_degree_sample.sum())
            # print('x_degree:', x_degree.shape, x_degree.max(), x_degree.min(), x_degree.mean())
            # x_tU = x_tU & (F.one_hot(x_start.long(), num_classes=2))
            
            # print('x_start_one:{}', x_start_one.shape, x_start_one.sum())
            # print('x_start_zero:{}', x_start_zero.shape, x_start_zero.sum())
            # import pdb
            # pdb.set_trace()


            out = self.p_mean_variance(model, x_start, t, x_tU=F.one_hot(x_start.long(), num_classes=2).float(), index=index, graph=x_start_ionh)
            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t
    
    def apply_noise(self, ts, x_start, x_base=None):
        # import time
        # time_s = time.time()

        batch_size, device = x_start.size(0), x_start.device
        tsF = ts.float() / batch_size
        Qtb = self.get_Qt_bar(tsF)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        # time1 = time.time()
        # print('time get Q:', time1 - time_s)
        Qtb = Qtb.to(device)
        # time2 = time.time()
        # print('Qt to', time2 - time1)
        # print('Qtb:{} x_start:{}'.format(Qtb.shape, x_start.shape))
        #print('Qtb:', Qtb.shape, Qtb.min(), Qtb.max(), Qtb.mean())
        # Compute transition probabilities
        # probX = x_start @ Qtb  # (bs, n, dx_out)
        #print('x_start:', x_start.shape)
        # import pdb
        # pdb.set_trace()
        # probX = x_start.unsqueeze(-1) * Qtb
        probX = x_start @ Qtb
        # time3 = time.time()
        # print('probX time:', time3 - time2)
        #print('probX:{}'.format(probX.shape))
        # print('probX:', probX.shape, probX.min(), probX.max(), probX.mean())
        # print('probX0:', probX[0].sum())
        # print('probX00:', probX[0,0].sum())
        # print('probX:{} probX2:{}'.format(probX.shape, probX2.shape))
        # import pdb
        # pdb.set_trace()
        

        sampled_t = self.sample_discrete_features(probX)

        # time4 = time.time()
        # print('sample time:', time4 - time3)
        #print('sampled_t:', sampled_t.shape)
        # discrete = random.randint(int(sampled_t.shape[1] * 0.8), sampled_t.shape[1])
        # x_t = th.where(sampled_t > discrete, 1.0, 0)
        x_t = F.one_hot(sampled_t, num_classes=2)

        # ct = x_t.argmax(dim=2)
        # print('x_t:{} ct:{} sampled_t:{}'.format(x_t.shape, ct.shape, sampled_t.shape))
        # print('x_start:', x_start.sum(), x_start.shape)
        # print('sampled_t:', sampled_t.sum(), sampled_t.shape)
        # dt = x_start.argmax(dim=2)
        # print('noise sum:', (dt==ct).sum())
        # import pdb
        # pdb.set_trace()

        # x_t = th.where(x_t > self.discrete, 1.0, 0)
        # result = torch.where(x > threshold, torch.tensor(1.), torch.tensor(0.))
        # print('noise:', noise.shape, noise.max(), noise.min(), noise.mean())
        #print('x_t:', x_t.shape, x_t.max(), x_t.min())
        # print('x_start:', x_start.shape, x_start.max(), x_start.min(), x_start.mean())
        # import pdb
        # pdb.set_trace()
        # if x_base is None:
        #     x_t = mix_tensors(x_start, x_t, 0.8)
        # else:
        #     x_t = mix_tensors(x_base, x_t, 0.99)
        return x_t
        
    
    def training_losses(self, model, x_start, reweight=False, index=None):
        # import time
        # time_s = time.time()

        if self.CatOneHot:
            #print('x_start:', x_start.shape)
            #print('x_startN:', x_start.sum(), x_start.shape)
            x_startU = F.one_hot(x_start.long(), num_classes=2)
            x_startU = x_startU.float()
            batch_size, device = x_startU.size(0), x_startU.device
            ts, pt = self.sample_timesteps(batch_size, device, 'importance')
            # time1 = time.time()
            # print('time sample:', time1 - time_s)
            if self.discrete_noise:
                x_tU = self.apply_noise(ts, x_startU)
                
                x_tU = x_tU & (F.one_hot(x_start.long(), num_classes=2))
                x_tU = x_tU.float()
                # print('x_tU:{} ts:{} x_startU:{}'.format(x_tU.shape, ts.shape, x_startU.shape))
                # print('equal per:', ((x_tU == x_startU).sum()) / (x_startU.shape[0] * x_startU.shape[1] * x_startU.shape[2]))
                # import pdb
                # pdb.set_trace()
            else:
                noise = th.randn_like(x_startU)
                if self.noise_scale != 0.:
                    x_tU = self.q_sample(x_startU, ts, noise)
                else:
                    x_tU = x_startU
            
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        # time1 = time.time()
        # print('time sample:', time1 - time_s)
        noise = th.randn_like(x_start)
        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        # print('eps per:', ((x_t - x_start < 0.01).sum()) / (x_start.shape[0] * x_start.shape[1]))
        # print('x_start:', x_start.shape)
        # print('ts:', tsF.shape)
        # time2 = time.time()
        # print('time add noise:', time2 - time1)
        # x_t = self.apply_noise(ts, x_start, x_base=x_t)
        # time3 = time.time()
        # print('time discrete noise:', time3 - time2)

        Closs = None
        terms = {}
        if self.CatOneHot:
            if self.indexIn:
                #model_output = model(x_t, ts, x_tU, index=index, graph=x_startU)
                # model_output, Closs = model(x_t, ts, x_tU, index=index, graph=x_startU, RCloss=True)
                # print('x_start:{} x_t:{} x_tU:{}'.format(x_start.shape, x_t.shape, x_tU.shape))
                # import pdb
                # pdb.set_trace()
                model_output, Closs = model(x_start, ts, F.one_hot(x_start.long(), num_classes=2).float(), index=index, graph=x_tU.long(), RCloss=True)
            else:
                model_output = model(x_start, ts, x_start)
                #Closs = 0
        else:
            model_output = model(x_start, ts)
        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        mse = mean_flat((target - model_output) ** 2)
        
        # if self.CatOneHot:
        #     print('x_t:{} x_tU:{}'.format(x_t.shape, x_tU.shape))
        #     x_tUd = x_tU.argmax(dim=2)
        #     print('x_tUd:', x_tUd.shape, (x_tUd == x_start.long()).sum())
            
        #     Closs = nt_xent_loss(x_t, x_tUd.float())
        #     print('Closs:{} mse:{}'.format(Closs, mse))
            
        #outputs = torch.sigmoid(model(inputs))
        #bce = self.criterion(th.sigmoid(model_output), target)
        # print('mse:{} bce:{}'.format(mse.mean(), bce.mean()))
        # import pdb
        # pdb.set_trace()
        #mse = bce * 0.05

        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = th.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / ((1-self.alphas_cumprod_prev[ts])**2 * (1-self.betas[ts]))
                weight = th.where((ts == 0), 1.0, weight)
                likelihood = mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output))**2 / 2.0)
                loss = th.where((ts == 0), likelihood, mse)
        else:
            weight = th.tensor([1.0] * len(target)).to(device)
        #print('weight:{} loss:{}'.format(weight, loss))
        terms["loss"] = weight * loss
        
        # update Lt_history & Lt_count
        for t, loss in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                #print('t:', loss.detach())
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                    self.Lt_count[t] += 1
                except:
                    #print(t)
                    #print(self.Lt_count[t])
                    #print(loss)
                    raise ValueError

        terms["loss"] /= pt
        if Closs is not None:
            terms["loss"] += Closs * 0.1
        # print('loss:{} Closs:{}'.format(terms['loss'].mean(), Closs))
        # import pdb
        # pdb.set_trace()
        return terms

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':  # importance sampling
            if not (self.Lt_count == self.history_num_per_term).all():
                #print('in:')
                return self.sample_timesteps(batch_size, device, method='uniform')
            
            Lt_sqrt = th.sqrt(th.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / th.sum(Lt_sqrt)
            pt_all *= 1- uniform_prob
            pt_all += uniform_prob / len(pt_all)
            #print('pt_all:', pt_all.shape, pt_all.sum(-1), self.Lt_history.sum())
            # import pdb
            # pdb.set_trace()
            assert pt_all.sum(-1) - 1. < 1e-5

            t = th.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt
        
        elif method == 'uniform':  # uniform sampling
            t = th.randint(0, self.steps, (batch_size,), device=device).long()
            pt = th.ones_like(t).float()

            return t, pt
            
        else:
            raise ValueError
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    

    def sample_discrete_features(self, probX):
        ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
            :param probX: bs, n, dx_out        node features
            :param probE: bs, n, n, de_out     edge features
            :param proby: bs, dy_out           global features.
        '''
        # import time
        # times = time.time()
        # print('sample begin:')
        bs, n, _ = probX.shape
        # print('probX:', probX.shape)
        # print('probX0', probX[0].sum())
        # Noise X
        # The masked rows should define probability distributions as well
        # probX[~node_mask] = 1 / probX.shape[-1]

        # Flatten the probability tensor to sample with multinomial
        probX = probX.reshape(bs * n, -1) #+ 1e-5   # (bs * n, dx_out)
        #print('probX0', probX[0].sum())
        #sums_per_row = probX.sum(dim=1, keepdim=True) + 1e-5
        
        
        # 确保求和不为零（避免除以零错误）
        # sums_per_row = th.where(sums_per_row == 0, th.ones_like(sums_per_row), sums_per_row)

        # 归一化，使每行之和为1
        #probX = probX / sums_per_row 
        # time1 = time.time()
        # print('time norm:', time1 - times)
        #print('probX:', probX.shape, probX.min(), probX.max(), probX.mean())
        # import pdb
        # pdb.set_trace()
        # Sample X
        X_t = probX.multinomial(1)    # (bs * n, 1)
        #print('x_t:', X_t.shape)
        # time2 = time.time()                              
        # print('time mul:', time2 - time1)
        X_t = X_t.reshape(bs, n)     # (bs, n)
        # import pdb
        # pdb.set_trace()
        return X_t
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x, t, x_tU=None, index=None, graph=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B, )
        if self.CatOneHot:
            if self.indexIn:
                model_output = model(x, t, x_tU, index=index, graph=graph)
            else:
                model_output = model(x, t, x_tU)
        else:
            model_output = model(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        
        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)
        
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
