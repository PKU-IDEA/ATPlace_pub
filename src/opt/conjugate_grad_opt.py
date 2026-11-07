import sys
import numpy as np 
import logging
import torch 
from torch.optim.optimizer import Optimizer

class CGOptimizer(Optimizer):
    """
    @brief Implement conjugate gradient descent optimization algorithm. 
    @param params variables, and learning rate  
    @param line_search_fn a callable line search function 
    """
    def __init__(self, params, line_search_fn=None):
        # g_k_1 is previous gradient, initial is zero  
        # d_k_1 is previous search direction, initial zero  
        defaults = dict(lr=1, g_k_1=[], d_k_1=[], line_search_fn=line_search_fn)
        super(CGOptimizer, self).__init__(params, defaults)
        
    def step(self):
        """
        @brief Performs a single optimization step.
        """
        for group in self.param_groups:           
            p = group['params'][0]
            g_k = p.grad.data
            #print("parameter before step:", p.data, g_k)
            if group['g_k_1'] == []: 
                group['g_k_1'] = torch.zeros_like(g_k)
                group['d_k_1'] = torch.zeros_like(g_k)
                g_k_1 = group['g_k_1']
                d_k_1 = group['d_k_1']
                beta_k = torch.zeros(1, dtype=g_k.dtype, device=g_k.device)
            else:
                g_k_1 = group['g_k_1']
                d_k_1 = group['d_k_1']
                # compute beta_k: g_k^T(g_k-g_k_1) / |g_k_1|_2^2
                beta_k = (g_k.dot(g_k.sub(g_k_1)).div(g_k_1.pow(2).sum()))

            # compute conjugate directions: d_k = -g_k + beta_k*d_k_1
            d_k = d_k_1.mul(beta_k).sub(g_k)
            # heuristic from NTUPlace for CG without line search 
            alpha_k = group['lr'].clone().detach().to(device=d_k.device) / d_k.norm()
            p.data.add_(alpha_k.mul(d_k))
            #print("alpha_k: ", alpha_k, p.data, p.grad.data, beta_k)
            
            g_k_1.data.copy_(g_k)
            d_k_1.data.copy_(d_k)
            #print("parameter after step:", p.data)

        return
