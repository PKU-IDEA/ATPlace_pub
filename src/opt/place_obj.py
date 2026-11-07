import os
import sys
import time
import math
import inspect
import itertools

import numpy as np
import torch 
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class PlaceObj(nn.Module):
    """
    @brief Define Placement objective:
        wirelength + density_weight * density penalty + temperature penalty
    """
    def __init__(self, loss_weight_init, params, data_collections, floorplan_params):
        """
        @brief initialize ops
        @param loss_weight: density and temperature weights in the objective
        @param params: parameters
        @param floorplan_params: floorplan parameters
        """
        super(PlaceObj, self).__init__()
        
        self.dtype, self.device = data_collections.pos[0].dtype, data_collections.pos[0].device
        self.data_collections = data_collections
        if len(loss_weight_init)==1:
            self.density_weight_init = loss_weight_init[0]
            self.density_weight = torch.tensor([self.density_weight_init], dtype=self.dtype, device=self.device)
        elif len(loss_weight_init)==2:
            self.density_weight_init, self.temp_weight_init = loss_weight_init
            self.density_weight = torch.tensor([self.density_weight_init], dtype=self.dtype, device=self.device)
            self.temp_weight = torch.tensor([self.temp_weight_init], dtype=self.dtype, device=self.device)
        else:
            raise ValueError(f"loss_weight_init {loss_weight_init} not valid!")
            
        self.num_bins_x, self.num_bins_y = data_collections.num_bins_x, data_collections.num_bins_y
        name = "%dx%d bins" % (self.num_bins_x, self.num_bins_y)

        self.base_gamma = params.gamma
        self.gamma = torch.tensor(self.base_gamma, dtype=self.dtype, device=self.device)
        self.eta = torch.tensor(params.eta, dtype=self.dtype, device=self.device)
        self.dis_bet_chips = params.dis_bet_chips
        self.theta_legal = torch.tensor([0, 0.25, 0.5, 0.75]).to(self.device) 
        # multiplied by pi*2 in triangular functions
        
        #self.update_gamma_op = self.update_gamma
        self.update_weight_op = self.build_update_weight(params)
        self.precondition_op = self.build_precondition(params)
        self.noise_op = self.build_noise(params)
        self.Liter_iteration = floorplan_params["iteration"]
        if 'Llambda_weight_iteration' in floorplan_params:
            self.Llambda_weight_iteration = floorplan_params['Llambda_weight_iteration']
        else:
            self.Llambda_weight_iteration = floorplan_params.get('density_weight_iterations', 1)
        if 'sub_iterations' in floorplan_params:
            self.sub_iterations = floorplan_params['sub_iterations']
        else:
            self.sub_iterations = 1
            
        #self.iteration = floorplan_params["iteration"]
        self.learning_rate = floorplan_params["learning_rate"]
        
    def compute_Rz(self, theta, epsilon = 0):
        '''
        Return the probalility of theta to be rotated to each legal orientation,
        including {0, pi/2, pi, pi*3/2}
        @param theta: the orientation of each chiplet
        '''
        tt = theta.view(-1,1).expand(theta.shape[0], self.theta_legal.shape[0])-self.theta_legal
        tt = 0.5-(0.5-tt.abs()).abs()
        Rz = torch.zeros((theta.shape[0], self.theta_legal.shape[0]), requires_grad=True).to(tt.device)
        legal_num = len(self.theta_legal)
        Rz = torch.where(tt<=1/2/legal_num, 1-2*legal_num**2*tt**2, Rz)
        Rz = torch.where((tt-1/2/legal_num)*(tt-1/legal_num)<0, 2*(legal_num*tt-1)**2, Rz)
        return Rz

    def wirelength_op(self, pos):
        '''
        Return the logsumexp wirelength, considering the orientation,
        following the ICCAD 23 paper: Orientation-aware SiP placement
        '''
        num_nodes = self.data_collections.num_nodes
        num_chiplets = self.data_collections.num_chiplets

        x = pos[0][:num_nodes].view(-1,1).expand(num_nodes, self.theta_legal.shape[0])
        y = pos[0][num_nodes:].view(-1,1).expand(num_nodes, self.theta_legal.shape[0])
        node_flip = self.data_collections.node_flip[:num_chiplets]        
        flip = node_flip.view(-1,1).expand(num_nodes, self.theta_legal.shape[0])
        theta = (pos[1]/torch.pi/2.0).type(torch.float32)

        Exp_Rz_coeff = (self.compute_Rz(theta)/self.eta).exp()
        Bz_coeff = Exp_Rz_coeff/Exp_Rz_coeff.sum(axis=-1,keepdim=True)
        filled_Bz_coeff = torch.cat(
            [Bz_coeff] + 
            [Bz_coeff.new(num_nodes-num_chiplets, self.theta_legal.shape[0]).fill_(
                1/self.theta_legal.shape[0]
            )]
        )

        cos_legal = torch.cos(self.theta_legal*torch.pi*2).reshape(1,-1)
        sin_legal = torch.sin(self.theta_legal*torch.pi*2).reshape(1,-1)
        #First calculate the possible length of all pins, then add them together in the loop
        x_pin_rotate = self.data_collections.pin_offset_x.reshape(-1,1)*cos_legal-\
                        self.data_collections.pin_offset_y.reshape(-1,1)*sin_legal
        y_pin_rotate = self.data_collections.pin_offset_x.reshape(-1,1)*sin_legal+\
                        self.data_collections.pin_offset_y.reshape(-1,1)*cos_legal
        
        pins = self.data_collections.pinid_2pin
        nodes = self.data_collections.pin2node_map[pins]
        sum_x = ((x[nodes, :] + (1-2*flip[nodes, :])*x_pin_rotate[pins]) * filled_Bz_coeff[nodes]).sum(dim=-1)
        sum_y = ((y[nodes, :] + y_pin_rotate[pins]) * filled_Bz_coeff[nodes]).sum(dim=-1)
        rmswl = (torch.diff(sum_x, dim=1).abs() + torch.diff(sum_y, dim=1).abs())
        hpwl = rmswl.sum()

        if hasattr(self.data_collections, "max_wirelength"):
            maxwl_term = torch.max((rmswl-self.data_collections.max_wirelength, 0))**2
            return hpwl, maxwl_term
        else:
            return hpwl

    def hpwl_op(self, pos):
        """
        @brief compute total HPWL
        @return hpwl of a net
        """
        num_nodes = self.data_collections.num_nodes
        num_chiplets = self.data_collections.num_chiplets
        pin_offset_x = self.data_collections.pin_offset_x
        pin_offset_y = self.data_collections.pin_offset_y
        hpwl = 0

        with torch.no_grad():
            
            x = pos[0][:num_nodes].view(-1,1)
            y = pos[0][num_nodes:].view(-1,1)
            theta = (pos[1]/torch.pi/2.0).view(-1,1).expand(num_chiplets, self.theta_legal.shape[0])
            tt = 0.5 - (0.5-(theta-self.theta_legal).abs()).abs()
            theta = self.theta_legal[torch.argmin(tt,axis=-1)].view(-1,1)
            theta = torch.cat([theta] + [theta.new(num_nodes-num_chiplets, 1).fill_(0)])    
            cos_legal = torch.round(torch.cos(theta*torch.pi*2)).view(-1,1)
            sin_legal = torch.round(torch.sin(theta*torch.pi*2)).view(-1,1)

            for net_id in range(len(self.data_collections.net2pin_map)):
                pins = self.data_collections.net2pin_map[net_id]
                nodes = self.data_collections.pin2node_map[pins]
                if len(torch.unique(nodes))<2:
                    continue

                x_pin_rotate = x[nodes].view(-1,1) + pin_offset_x[pins].reshape(-1,1)*cos_legal[nodes]-\
                                pin_offset_y[pins].reshape(-1,1)*sin_legal[nodes]
                hpwl_x = torch.amax(x_pin_rotate, dim=0) - torch.amin(x_pin_rotate, dim=0)

                y_pin_rotate = y[nodes].view(-1,1) + pin_offset_x[pins].reshape(-1,1)*sin_legal[nodes]+\
                                pin_offset_y[pins].reshape(-1,1)*cos_legal[nodes]
                hpwl_y = torch.amax(y_pin_rotate, dim=0) - torch.amin(y_pin_rotate, dim=0)

                #hpwl = max(hpwl, (hpwl_x**2+hpwl_y**2)**0.5)
                hpwl += (hpwl_x+hpwl_y)

        #sum_x = ((x[nodes,:]+x_pin_rotate[pins])*filled_Bz_coeff[nodes]).sum(axis=-1)
        #hpwl_x = torch.logsumexp(sum_x/self.gamma, 1) + torch.logsumexp(-sum_x / self.gamma, 1)
        #lsewl = self.gamma * (hpwl_x + hpwl_y).sum()
        #maxwl = self.gamma * torch.logsumexp(rmswl/self.gamma, 0)
                
        return hpwl
    
    def density_op(self, pos):
        bin_size_x = self.data_collections.bin_size_x
        bin_size_y = self.data_collections.bin_size_y
        target_dens = self.data_collections.target_dens
        return ((self.density_bell(pos)-target_dens*bin_size_x*bin_size_y)**2).sum()#.relu_()

    def density_bell(self, pos):
        '''
        Return the density of bins, calculated by the bell-shaped function
        '''
        num_nodes = self.data_collections.num_nodes
        num_chiplets = self.data_collections.num_chiplets
        num_bins_x = self.data_collections.num_bins_x
        num_bins_y = self.data_collections.num_bins_y
        bin_size_x = self.data_collections.bin_size_x
        bin_size_y = self.data_collections.bin_size_y

        cos_legal = torch.cos(self.theta_legal*torch.pi*2).reshape(1,-1)
        sin_legal = torch.sin(self.theta_legal*torch.pi*2).reshape(1,-1)
        node_size_x = self.data_collections.node_size_x[:num_chiplets].reshape(-1,1)+self.dis_bet_chips
        node_size_y = self.data_collections.node_size_y[:num_chiplets].reshape(-1,1)+self.dis_bet_chips

        x = pos[0][:num_chiplets]
        y = pos[0][num_nodes:num_nodes+num_chiplets]
        theta = (pos[1]/torch.pi/2.0).type(torch.float32)
        #Exp_Rz_coeff = (self.compute_Rz(theta)/self.eta).exp()
        #Bz_coeff = Exp_Rz_coeff/Exp_Rz_coeff.sum(-1).reshape(-1,1)
        Bz_coeff = torch.softmax(self.compute_Rz(theta) / self.eta, dim=-1)
        num_orients = Bz_coeff.shape[-1]
        size_x_new = (node_size_x*cos_legal + node_size_y*sin_legal).abs().unsqueeze(-1)
        size_y_new = (node_size_x*sin_legal + node_size_y*cos_legal).abs().unsqueeze(-1)

        a_x = (4/(size_x_new+2*bin_size_x)/(size_x_new+4*bin_size_x)).to(self.device)
        b_x = 2/bin_size_x/(size_x_new+4*bin_size_x).to(self.device) #Shape [num_nodes, num_orients, 1]
        px = torch.zeros(num_chiplets,num_orients,num_bins_x).to(self.device)
        dx = (self.data_collections.bin_center_x.to(self.device).reshape(1,1,-1) - x.reshape(-1,1,1)).abs()
        px = torch.where(dx<=(size_x_new/2+bin_size_x), 1-a_x*dx*dx, px)
        px = torch.where((dx-(size_x_new/2+bin_size_x))*(dx-(size_x_new/2+2*bin_size_x))<0, 
                         b_x*(dx-(size_x_new/2+2*bin_size_x))**2, px)
        px = px.unsqueeze(-1).repeat(1,1,1,num_bins_y)

        a_y = (4/(size_y_new+2*bin_size_y)/(size_y_new+4*bin_size_y)).to(self.device)
        b_y = 2/bin_size_y/(size_y_new+4*bin_size_y).to(self.device)
        py = torch.zeros(num_chiplets,num_orients,num_bins_y).to(self.device)
        dy = (self.data_collections.bin_center_y.to(self.device).reshape(1,1,-1) - y.reshape(-1,1,1)).abs()
        py = torch.where(dy<=(size_y_new/2+bin_size_y), 1-a_y*dy*dy, py)
        py = torch.where((dy-(size_y_new/2+bin_size_y))*(dy-(size_y_new/2+2*bin_size_y))<0, 
                         b_y*(dy-(size_y_new/2+2*bin_size_y))**2, py)
        py = py.unsqueeze(-2).repeat(1,1,num_bins_x, 1)

        density = (px*py*Bz_coeff.reshape(num_chiplets,num_orients,1,1)).sum(1)
        c_v = ((node_size_x*node_size_y).reshape(-1,1,1)/density.sum((-1,-2),keepdim=True)).detach()

        return (c_v*density).sum(0)
    
    def density_bell(self, pos):
        num_nodes = self.data_collections.num_nodes
        num_chiplets = self.data_collections.num_chiplets
        num_bins_x = self.data_collections.num_bins_x
        num_bins_y = self.data_collections.num_bins_y
        bin_size_x = self.data_collections.bin_size_x
        bin_size_y = self.data_collections.bin_size_y

        cos_legal = torch.cos(self.theta_legal * torch.pi * 2).reshape(1, -1)
        sin_legal = torch.sin(self.theta_legal * torch.pi * 2).reshape(1, -1)
        node_size_x = self.data_collections.node_size_x[:num_chiplets].reshape(-1, 1) + self.dis_bet_chips
        node_size_y = self.data_collections.node_size_y[:num_chiplets].reshape(-1, 1) + self.dis_bet_chips

        x = pos[0][:num_chiplets]
        y = pos[0][num_nodes:num_nodes + num_chiplets]
        theta = (pos[1] / torch.pi / 2.0).type(torch.float32)

        Rz = self.compute_Rz(theta)
        Bz_coeff = torch.softmax(Rz / self.eta, dim=-1)
        num_orients = Bz_coeff.shape[-1]
        size_x_new = (node_size_x * cos_legal + node_size_y * sin_legal).abs().unsqueeze(-1)
        size_y_new = (node_size_x * sin_legal + node_size_y * cos_legal).abs().unsqueeze(-1)

        eps = 1e-8
        a_x = 4.0 / (size_x_new + 2 * bin_size_x + eps) / (size_x_new + 4 * bin_size_x + eps)
        b_x = 2.0 / (bin_size_x + eps) / (size_x_new + 4 * bin_size_x + eps)
        px = torch.zeros(num_chiplets, num_orients, num_bins_x, device=self.device)
        dx = (self.data_collections.bin_center_x.to(self.device).reshape(1, 1, -1) - x.reshape(-1, 1, 1)).abs()
        px = torch.where(dx <= (size_x_new / 2 + bin_size_x), 1 - a_x * dx * dx, px)
        px = torch.where((dx - (size_x_new / 2 + bin_size_x)) * (dx - (size_x_new / 2 + 2 * bin_size_x)) < 0,
                         b_x * (dx - (size_x_new / 2 + 2 * bin_size_x)) ** 2, px)
        px = px.unsqueeze(-1).repeat(1, 1, 1, num_bins_y)

        a_y = 4.0 / (size_y_new + 2 * bin_size_y + eps) / (size_y_new + 4 * bin_size_y + eps)
        b_y = 2.0 / (bin_size_y + eps) / (size_y_new + 4 * bin_size_y + eps)
        py = torch.zeros(num_chiplets, num_orients, num_bins_y, device=self.device)
        dy = (self.data_collections.bin_center_y.to(self.device).reshape(1, 1, -1) - y.reshape(-1, 1, 1)).abs()
        py = torch.where(dy <= (size_y_new / 2 + bin_size_y), 1 - a_y * dy * dy, py)
        py = torch.where((dy - (size_y_new / 2 + bin_size_y)) * (dy - (size_y_new / 2 + 2 * bin_size_y)) < 0,
                         b_y * (dy - (size_y_new / 2 + 2 * bin_size_y)) ** 2, py)
        py = py.unsqueeze(-2).repeat(1, 1, num_bins_x, 1)

        density = (px * py * Bz_coeff.reshape(num_chiplets, num_orients, 1, 1)).sum(1)
        area = (node_size_x * node_size_y).reshape(-1, 1, 1)
        density_sum = density.sum((-1, -2), keepdim=True)
        c_v = (area / (density_sum + eps)).detach()
        result = (c_v * density).sum(0)

        if not torch.isfinite(result).all():
            print("NaN in density_bell")
            if not torch.isfinite(Rz).all(): print("Rz NaN")
            if not torch.isfinite(Bz_coeff).all(): print("Bz_coeff NaN")
            if not torch.isfinite(px).all(): print("px NaN")
            if not torch.isfinite(py).all(): print("py NaN")
            if not torch.isfinite(density).all(): print("density NaN")
            if not torch.isfinite(c_v).all(): print("c_v NaN")

        return result

    def density_sigmoid(self, pos):
        '''
        Return the density of bins, calculated by the 2nd order sigmoid function
        '''
        num_nodes = self.data_collections.num_nodes
        num_chiplets = self.data_collections.num_chiplets
        num_bins_x = self.data_collections.num_bins_x
        num_bins_y = self.data_collections.num_bins_y
        bin_size_x = self.data_collections.bin_size_x
        bin_size_y = self.data_collections.bin_size_y

        cos_legal = torch.cos(self.theta_legal*torch.pi*2).reshape(1,-1)
        sin_legal = torch.sin(self.theta_legal*torch.pi*2).reshape(1,-1)
        node_size_x = self.data_collections.node_size_x[:num_chiplets].reshape(-1,1)+self.dis_bet_chips
        node_size_y = self.data_collections.node_size_y[:num_chiplets].reshape(-1,1)+self.dis_bet_chips

        x = pos[0][:num_chiplets]
        y = pos[0][num_nodes:num_nodes+num_chiplets]
        theta = (pos[1]/torch.pi/2.0).type(torch.float32)
        Exp_Rz_coeff = (self.compute_Rz(theta)/self.eta).exp()
        Bz_coeff = Exp_Rz_coeff/Exp_Rz_coeff.sum(-1,keepdim=True)
        num_orients = Bz_coeff.shape[-1]
        size_x_new = (node_size_x*cos_legal + node_size_y*sin_legal).abs().unsqueeze(-1)
        size_y_new = (node_size_x*sin_legal + node_size_y*cos_legal).abs().unsqueeze(-1)

        dx = (self.data_collections.bin_center_x.to(self.device).reshape(1,1,-1) - x.reshape(-1,1,1)).abs()
        px = torch.zeros(num_chiplets,num_orients,num_bins_x).to(self.device)
        px = torch.where((dx-size_x_new/2)*(dx-(size_x_new+bin_size_x)/2)<0,
                         2/bin_size_x**2*(dx-(size_x_new+bin_size_x)/2)**2, px)
        px = torch.where((dx-size_x_new/2)*(dx-(size_x_new-bin_size_x)/2)<0, 
                         1-2/bin_size_x**2*(dx-(size_x_new-bin_size_x)/2)**2, px)
        px = torch.where(dx<(size_x_new-bin_size_x)/2, 1, px) 
        px = px.unsqueeze(-1).repeat(1,1,1,num_bins_y)

        dy = (self.data_collections.bin_center_y.to(self.device).reshape(1,1,-1) - y.reshape(-1,1,1)).abs()
        py = torch.zeros(num_chiplets,num_orients,num_bins_y).to(self.device)
        py = torch.where((dy-size_y_new/2)*(dy-(size_y_new+bin_size_y)/2)<0,
                         2/bin_size_y**2*(dy-(size_y_new+bin_size_y)/2)**2, py)
        py = torch.where((dy-size_y_new/2)*(dy-(size_y_new-bin_size_y)/2)<0, 
                         1-2/bin_size_y**2*(dy-(size_y_new-bin_size_y)/2)**2, py)
        py = torch.where(dy<(size_y_new-bin_size_y)/2, 1, py) 
        py = py.unsqueeze(-2).repeat(1,1,num_bins_x, 1)

        density = (px*py*Bz_coeff.reshape(num_chiplets,num_orients,1,1)).sum(1)
        c_v = bin_size_x*bin_size_y
        #((node_size_x*node_size_y).reshape(-1,1,1)/density.sum((-1,-2),keepdim=True)).detach()

        return (c_v*density).sum(0)
    
    def temperature_term(self, Temp):
        temp_threshold = self.data_collections.temp_threshold
        #temp_grad_th = self.data_collections.temp_grad_threshold
        Temp_loss = (Temp-temp_threshold).pow(self.gamma).sum()
        return Temp_loss
    
    
    def physics_op(self, pos):
        num_nodes = self.data_collections.num_nodes
        num_chiplets = self.data_collections.num_chiplets
        node_size_x = self.data_collections.node_size_x[:num_chiplets].reshape(-1,1) 
        node_size_y = self.data_collections.node_size_y[:num_chiplets].reshape(-1,1)
        W, H = self.data_collections.intp_width, self.data_collections.intp_height
        num_grid_x, num_grid_y = self.data_collections.num_grid_x, self.data_collections.num_grid_y
        
        x = pos[0][:num_chiplets]
        y = pos[0][num_nodes:num_nodes+num_chiplets]
        theta = (pos[1]/torch.pi/2.0).type(torch.float32)
        Exp_Rz_coeff = (self.compute_Rz(theta)/self.eta).exp()
        Bz_coeff = Exp_Rz_coeff/Exp_Rz_coeff.sum(axis=-1).reshape(-1,1)
        cos_legal = torch.cos(self.theta_legal*torch.pi*2).reshape(1,-1)
        sin_legal = torch.sin(self.theta_legal*torch.pi*2).reshape(1,-1)
        size_x_new = ((node_size_x*cos_legal - node_size_y*sin_legal).abs() * Bz_coeff).sum(axis=-1)
        size_y_new = ((node_size_x*sin_legal + node_size_y*cos_legal).abs() * Bz_coeff).sum(axis=-1)
        
        Temp = self.temp_model((x, y, size_x_new, size_y_new, self.powermap))
        fields = {'Thermal': Temp + self.amb}
        return fields
        
    def obj_fn(self, pos):
        """
        @brief Compute objective.
            wirelength + density_weight * density penalty + temp_weight * temperature penalty
        @param pos locations of cells
        @return objective value
        """
        wirelength = self.wirelength_op(pos)
        density = self.density_op(pos)
        loss = wirelength + self.density_weight*density
        if self.temp_aware_opt:
            physical_field = self.physics_op(pos)
            temp_penalty = self.temperature_term(physical_field['Thermal'])
            loss += self.temp_weight*temp_penalty
        return loss

    def obj_and_grad_fn(self, pos):
        """
        @brief compute objective and gradient.
            wirelength + density_weight * density penalty + temp_weight * temperature penalty
        @param pos: locations of cells
        @return objective value
        
        density.backward()
        #print("Before precondition_op:", pos[1].data, pos[1].grad)
        self.precondition_op([pos[0].grad, pos[1].grad])
        print("Densgrad: ", pos[0].data, pos[0].grad, pos[1].data, pos[1].grad)

        """
        
        obj = self.obj_fn(pos)
        
        if pos[0].grad is not None:
            pos[0].grad.zero_()
        if pos[1].grad is not None:
            pos[1].grad.zero_()

        obj.backward()
        self.precondition_op([pos[0].grad, pos[1].grad])
        
        return obj, [pos[0].grad, pos[1].grad]

    def forward(self):
        """
        @brief Compute objective with current locations of cells.
        """
        return self.obj_fn(self.pos)
    
    def estimate_initial_lr(self, pos, lr):
        """
        @brief Estimate initial learning rate by moving a small step.
        Computed as | x_k - x_k_1 |_2 / | g_k - g_k_1 |_2.
        @param x_k current solution
        @param lr small step
        """
        lr_pos, lr_ang = lr
        val0, g_0 = self.obj_and_grad_fn(pos)
        
        pos_pos = torch.autograd.Variable(pos[0] - lr_pos * g_0[0], requires_grad=True)
        val_pos, g_pos = self.obj_and_grad_fn([pos_pos, pos[1]])

        pos_ang = torch.autograd.Variable(pos[1] - lr_ang * g_0[1], requires_grad=True)
        val_ang, g_ang = self.obj_and_grad_fn([pos[0], pos_ang])

        return [(pos[0] - pos_pos).norm(p=2) / (g_0[0] - g_pos[0]).norm(p=2), 
                (pos[1] - pos_ang).norm(p=2) / (g_0[1] - g_ang[1]).norm(p=2)]

    def initialize_weight(self, pos, temp_aware_opt):
        """
        @brief compute initial density/temp weights using gradient norms
        """
        # Helper: compute L1 grad norm w.r.t. pos[0] for a scalar loss
        def compute_grad_norm(loss_tensor, inputs):
            grad_outputs = torch.ones_like(loss_tensor)
            grads = torch.autograd.grad(
                outputs=loss_tensor,
                inputs=inputs,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False
            )
            return grads[0].norm(p=1)

        if pos[0].grad is not None:
            pos[0].grad.zero_()
        if pos[1].grad is not None:
            pos[1].grad.zero_()
            
        # --- Wirelength gradient ---
        wirelength = self.wirelength_op(pos)
        wirelength_grad_norm_pos = compute_grad_norm(wirelength, [pos[0], pos[1]])
        print("Pos: ", pos[0].norm(p=1), pos[1].norm(p=1))
        print("WLgrad: ", wirelength_grad_norm_pos)

        # --- Density gradient ---
        density = self.density_op(pos)
        density_grad_norm_pos = compute_grad_norm(density, [pos[0], pos[1]])
        print("Densgrad: ", density.norm(p=1), density_grad_norm_pos)
        self.density_weight = torch.tensor(
            [self.density_weight_init * wirelength_grad_norm_pos / density_grad_norm_pos], 
            dtype=self.dtype, device=self.device
        )

        if temp_aware_opt:
            # --- Thermal gradient ---
            physical_field = self.physics_op(pos)
            temp = physical_field['Thermal'].sum()
            temp_grad_norm_pos = compute_grad_norm(temp, [pos[0], pos[1]])
            print("Densgrad: ", self.temp_weight_init, temp_grad_norm_pos)
            self.temp_weight = torch.tensor(
                [self.temp_weight_init * wirelength_grad_norm_pos / temp_grad_norm_pos], 
                dtype=self.dtype, device=self.device
            )

    
    def build_update_weight(self, params):
        """
        @brief update density weight
        @param params parameters
        """
        ref_hpwl = params.ref_hpwl
        LOWER_PCOF = torch.Tensor([params.density_weight_lower_ratio])
        UPPER_PCOF = torch.Tensor([params.density_weight_upper_ratio])
        
        def update_weight_op(iteration, cur_metric, prev_metric):
            ### based on hpwl
            with torch.no_grad():
                delta_hpwl = cur_metric.hpwl - prev_metric.hpwl
                mu = torch.pow(UPPER_PCOF, 1-delta_hpwl/ref_hpwl).clamp(min=LOWER_PCOF, max=UPPER_PCOF)
            return mu.item()

        return update_weight_op

    def build_noise(self, params,):
        """
        @brief add noise to chiplet locations
        @param params: parameters
        """
        num_nodes = self.data_collections.num_nodes
        num_chiplets = self.data_collections.num_chiplets
        intp_size = torch.concatenate([
                (self.data_collections.xhigh-self.data_collections.xlow)*torch.ones(num_nodes), 
                (self.data_collections.yhigh-self.data_collections.ylow)*torch.ones(num_nodes)], axis=0)

        def noise_op(pos, pertub_size, percent):
            indices = torch.randperm(num_chiplets)[:math.ceil(num_chiplets*percent)]
            mask = torch.zeros(num_chiplets).to(pos[0].device)    
            mask[indices] = 1
            
            with torch.no_grad():
                noise = torch.rand(num_nodes*2)
                noise.sub_(0.5).mul_(intp_size*pertub_size)
                # no noise to terminals
                noise[:num_nodes].mul_(mask)
                #noise[num_chiplets:num_nodes].zero_()
                noise[num_nodes:].mul_(mask)
                #noise[num_nodes+num_chiplets:num_nodes*2].zero_()
                pos[0].add_(noise)

                noise = torch.rand_like(pos[1])
                noise.sub_(0.5).mul_(torch.pi/2)
                noise[num_chiplets:num_nodes].zero_()
                pos[1].add_(noise)

                self.move_boundary_op(pos)
            return
                
        return noise_op
    
    def build_precondition(self, params):
        """
        @brief preconditioning to gradient
        @param params: parameters
        """
        num_nodes = self.data_collections.num_nodes
        num_chiplets = self.data_collections.num_chiplets
        num_pins_in_nodes = np.zeros(self.data_collections.num_nodes)
        sum_pin_locs_in_nodes = np.zeros(self.data_collections.num_nodes)
        node2pin_map = self.data_collections.node2pin_map
        pin_offset_x = self.data_collections.pin_offset_x
        pin_offset_y = self.data_collections.pin_offset_y
        
        num_pins_in_nodes = torch.tensor(num_pins_in_nodes, dtype=self.dtype, device=self.device)
        precond = (num_pins_in_nodes + self.density_weight*self.data_collections.node_size)[:num_chiplets]
        precond.clamp_(min=1.0)#.clamp_(max=100.0)
        sum_value = (pin_offset_x.abs().sum() + pin_offset_y.abs().sum()).clone().detach()
        # With all pins at their centers, use neutral scaling instead of dividing by zero.
        sum_value = torch.where(sum_value > 0, sum_value, torch.ones_like(sum_value))
        precond_ang = sum_value.to(dtype=self.dtype, device=self.device)
        
        def precondition_op(grad):
            grad_pos, grad_ang = grad
            #grad_pos[0:num_chiplets].div_(precond)
            grad_pos[num_chiplets:num_nodes].zero_()
            #grad_pos[num_nodes:num_nodes+num_chiplets].div_(precond)
            grad_pos[num_nodes+num_chiplets:num_nodes*2].zero_()
            
            grad_ang.div_(precond_ang)

            return [grad_pos, grad_ang]

        return precondition_op
    
    def legalize_theta(self, theta, dis_bet_chips=0):
        num_nodes = self.data_collections.num_nodes
        num_chiplets = self.data_collections.num_chiplets
        node_size_x = self.data_collections.node_size_x[:num_chiplets] + dis_bet_chips
        node_size_y = self.data_collections.node_size_y[:num_chiplets] + dis_bet_chips
        
        theta = (theta/torch.pi/2).frac()
        theta = (theta-torch.floor(theta)+1).frac()
        tt = theta.view(-1,1).expand(-1, self.theta_legal.shape[0])-self.theta_legal
        tt = (0.5-(0.5-tt).abs()).abs()
        theta_legal = self.theta_legal[torch.argmin(tt,axis=-1)].view(-1)
        #size_x_new = torch.abs(node_size_x*torch.cos(theta_legal)-node_size_y*torch.sin(theta_legal))
        #size_y_new = torch.abs(node_size_x*torch.sin(theta_legal)+node_size_y*torch.cos(theta_legal))
        
        size_x_new, size_y_new = node_size_x.clone(), node_size_y.clone()
        size_x_new = torch.where((theta-0.125)*(theta-0.375)<0,node_size_y, size_x_new)
        size_x_new = torch.where((theta-0.625)*(theta-0.875)<0,node_size_y, size_x_new)
        size_y_new = torch.where((theta-0.125)*(theta-0.375)<0,node_size_x, size_y_new)
        size_y_new = torch.where((theta-0.625)*(theta-0.875)<0,node_size_x, size_y_new)

        return theta*torch.pi*2, theta_legal*torch.pi*2, size_x_new, size_y_new
        
    def move_boundary_op(self, pos):
        num_nodes = self.data_collections.num_nodes
        num_chiplets = self.data_collections.num_chiplets
        xlow, ylow = self.data_collections.xlow, self.data_collections.ylow
        xhigh, yhigh = self.data_collections.xhigh, self.data_collections.yhigh

        with torch.no_grad():
            x = pos[0].data[:num_chiplets].clone()
            y = pos[0].data[num_nodes:num_nodes+num_chiplets].clone()
            #print("y_old data:", 0.1*torch.round(10*pos[0].data[num_nodes:num_nodes+num_chiplets+1].data)) 
            
            theta, _, size_x_new, size_y_new = self.legalize_theta(pos[1].data.clone())

            x = torch.where(x < xlow+size_x_new/2, xlow+size_x_new/2, x)
            x = torch.where(x > xhigh-size_x_new/2, xhigh-size_x_new/2, x)
            y = torch.where(y < ylow+size_y_new/2, ylow+size_y_new/2, y)
            y = torch.where(y > yhigh-size_y_new/2, yhigh-size_y_new/2, y)
            
            pos[0].data[:num_chiplets] = x
            pos[0].data[num_nodes:num_nodes+num_chiplets] = y
            pos[1].data = theta
