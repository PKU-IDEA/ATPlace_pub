import os 
import sys
import time 
import math
import logging

import numpy as np 
import torch 
import torch.nn as nn
from tqdm import trange

from utils.visualize_temp import plot_fp

from . import eval_metrics
from . import conjugate_grad_opt
from .place_obj import PlaceObj


class DataCollection(object):
    """
    @brief A wraper for all data tensors on device for building ops
    """
    def __init__(self, pos, params, system, device):
        """
        @brief initialization
        @param pos locations of cells
        @param params parameters
        @param system placement database
        @param device cpu or cuda
        """
        self.device = device
        torch.set_num_threads(params.num_threads)
        # position should be parameter
        self.pos = pos
        self.num_nodes = system.num_nodes
        self.num_chiplets = system.num_chiplets
        self.num_terminals = system.num_terminals
        self.xlow, self.ylow = system.xlow, system.ylow
        self.xhigh, self.yhigh =  system.xhigh, system.yhigh
        self.intp_width, self.intp_height = system.intp_width, system.intp_height
        self.num_grid_x, self.num_grid_y = system.num_grid_x, system.num_grid_y
        self.total_chip_area = system.node_area.sum()
        
        with torch.no_grad():
            # other tensors required to build ops

            self.node_size_x = torch.from_numpy(system.node_size_x).to(device)
            self.node_size_y = torch.from_numpy(system.node_size_y).to(device)
            self.node_flip = torch.Tensor(system.node_flip).to(device)

            self.pin_offset_x = torch.tensor(system.pin_offset_x, dtype=self.pos[0].dtype, device=device)
            self.pin_offset_y = torch.tensor(system.pin_offset_y, dtype=self.pos[0].dtype, device=device)
            self.dis_bet_chips = params.dis_bet_chips

            self.tot_mov_areas = ((self.node_size_x+self.dis_bet_chips)*(self.node_size_y+self.dis_bet_chips)).sum()
            self.node_size = self.node_size_x*self.node_size_y

            self.pin2node_map = torch.from_numpy(system.pin2node_map).to(device)
            self.node2pin_map = system.node2pin_map
            # number of pins for each cell
            self.pin2net_map = torch.from_numpy(system.pin2net_map).to(device)
            self.net2pin_map = system.net2pin_map
            if np.amin(system.net_weights) != np.amax(system.net_weights):  # weights are meaningful
                self.net_weights = torch.from_numpy(system.net_weights).to(device)
            else:  # an empty tensor
                logging.debug("net weights are all the same, ignored")
                self.net_weights = torch.Tensor().to(device)

            self.netid_2pin = torch.from_numpy(system.netid_2pin).to(device)
            self.pinid_2pin = torch.from_numpy(system.pinid_2pin).to(device)
            self.netid_multipin = torch.from_numpy(system.netid_multipin).to(device)

            # number of pins for each node
            num_pins_in_nodes = np.zeros(system.num_nodes)
            for i in range(system.num_nodes):
                num_pins_in_nodes[i] = len(system.node2pin_map[i])
            self.num_pins_in_nodes = torch.tensor(num_pins_in_nodes, dtype=self.pos[0].dtype, device=device)

            self.num_bins_x = system.num_bins_x
            self.num_bins_y = system.num_bins_y
            self.bin_size_x = system.bin_size_x
            self.bin_size_y = system.bin_size_y
            self.bin_center_x = torch.from_numpy(system.bin_center_x).to(device)
            self.bin_center_y = torch.from_numpy(system.bin_center_y).to(device)
            avg_dens = self.tot_mov_areas/(self.xhigh-self.xlow)/(self.yhigh-self.ylow)
            self.target_dens = avg_dens.item()  # Fixed to the actual average occupied density.
            logging.info(f"target density is the average occupied density: {avg_dens.item():.2f}.")
            self.temp_threshold = params.temp_threshold

            
class NonlinearPlace(nn.Module):
    # Basic nonlinear placement optimization class

    def __init__(self, params, system):
        super(NonlinearPlace, self).__init__()
        torch.manual_seed(params.random_seed)

        tt = time.time()
        self.device = torch.device("cpu")
        self.init_pos = np.zeros(system.num_nodes*2, dtype=system.dtype)
        self.init_angle = np.zeros(system.num_chiplets, dtype=system.dtype)
        self.init_pos[:system.num_nodes] = system.node_x
        self.init_pos[system.num_nodes:] = system.node_y
        self.init_angle = system.node_orient[:system.num_chiplets]

        # Preserve the ILP initialization; random center initialization is disabled.

        # position and orientation of chiplets should be parameter 
        self.pos = nn.ParameterList([nn.Parameter(torch.from_numpy(self.init_pos).to(self.device)), 
                                     nn.Parameter(torch.from_numpy(self.init_angle).to(self.device))])
        self.datacollection = DataCollection(self.pos, params, system, self.device)
        #logging.info("preparing floorplan takes %.2f seconds" % (time.time()-tt))
    
    def Llambda_stop_criterion(self, Llambda_weight_step, metrics):
        with torch.no_grad():
            if len(metrics) > 1:
                cur_metric = metrics[-1][-1]
                prev_metric = metrics[-2][-1]
                if (cur_metric.hpwl > prev_metric.hpwl and 
                    cur_metric.overflow < self.stop_overflow): 
                    logging.debug("Llambda stopping when %d and ( %g < 0.05 and %g > %g )" 
                                  %(Llambda_weight_step, cur_metric.overflow, 
                                    cur_metric.hpwl, prev_metric.hpwl))
                    return True
        return False
    
    def Lsub_stop_criterion(self, Llambda_weight_step, Lsub_step, metrics):
        moving_avg_window = self.moving_avg_window
        with torch.no_grad():
            if len(metrics) >= moving_avg_window * 2:
                cur_avg_obj = 0
                prev_avg_obj = 0
                for i in range(moving_avg_window):
                    cur_avg_obj += metrics[-1-i].objective
                    prev_avg_obj += metrics[-1-moving_avg_window-i].objective
                cur_avg_obj /= moving_avg_window
                prev_avg_obj /= moving_avg_window
                threshold = 0.999
                if cur_avg_obj >= prev_avg_obj * threshold:
                    logging.debug("Lsub stopping when %d and %g > %g * %g"
                                  %(Lsub_step, cur_avg_obj, prev_avg_obj, threshold))
                    return True
        return False

    def check_plateau(self, x, window=10, threshold=1e-3):
        if len(x) < window:
            return False
        x = np.array(x[-window:])[:,0]
        if (np.max(x) - np.min(x)) / np.mean(x) < threshold:
            logging.debug(f"Wirelength plateau detected, ratio = {(np.max(x)-np.min(x))/np.mean(x):.5f}.")
            return True
        return False
    
    def check_divergence(self, x, best_metric, window=50, threshold=0.1):
        ## divergence threshold should decrease as overflow decreases
        ## only detect divergence when overflow is relatively low but not too low
        if len(x) < window or best_metric[0] is None:
            return False
        x = np.array(x[-window:])
        overflow_mean = np.mean(x[:, 1])
        overflow_diff = np.maximum(0, np.sign(x[1:, 1] - x[:-1, 1])).astype(np.float32)
        overflow_diff = np.sum(overflow_diff) / overflow_diff.shape[0]
        overflow_range = np.max(x[:, 1]) - np.min(x[:, 1])
        wl_mean = np.mean(x[:, 0])
        wl_ratio = wl_mean/best_metric[0].hpwl - 1
        overflow_ratio = (overflow_mean - max(self.stop_overflow, 
                                              best_metric[0].overflow))/best_metric[0].overflow
        if overflow_range/overflow_mean < 1e-3:
            logging.debug(f"OVFL plateau detected, ratio {overflow_range/overflow_mean:.5f}")
            return True
        if wl_ratio > threshold:
            if overflow_ratio > threshold:
                logging.debug(f"OVFL Divergence: overflow increases too much, {overflow_mean:.4f} "+\
                            f"> {best_metric[0].overflow:.4f} and {overflow_ratio:.4f} > {threshold:.4f}")
                return True
            elif overflow_diff > 0.6:
                logging.debug("OVFL fluctuates too frequently")
                return False
        return False


    def __call__(self, params, system):
        """
        @brief Top API to solve floorplan.
        @param params: parameters
        @param system: floorplan database
        """
        all_metrics = []
        num_chiplets = system.num_chiplets
        num_nodes = system.num_nodes

        # The public flow constructs exactly one placement stage.
        for floorplan_params in params.floorplan_stages:

            def initialize_learning_rate(system):
                #lr_init = model.estimate_initial_lr(self.pos, [lr_params, lr_params])
                for idx in range(len(optimizer.param_groups)):
                    if len(floorplan_params["learning_rate"])==1:
                        optimizer.param_groups[idx]["lr"] = torch.tensor(
                            floorplan_params["learning_rate"], dtype=torch.float64
                        )
                    else:
                        optimizer.param_groups[idx]["lr"] = torch.tensor(
                            floorplan_params["learning_rate"][idx], dtype=torch.float64
                        )
            
            def one_descent_step(Liter_step, Llambda_weight_step, Lsub_step, iteration, metrics):
                t0 = time.time()

                cur_metric = eval_metrics.EvalMetrics(
                    model, iteration, (Liter_step, Llambda_weight_step, Lsub_step)
                )
                if self.temp_aware_opt:
                    cur_metric.temp_threshold = model.temp_threshold
                    cur_metric.temp_weight = model.temp_weight.data

                metrics.append(cur_metric)
                pos = self.pos
                optimizer.zero_grad(set_to_none=False)

                t1 = time.time()
                obj, grad = model.obj_and_grad_fn(pos)
                cur_metric.objective = obj.data.clone().item()
                cur_metric.gradnorm = [grad[0].norm(p=1).item(), grad[1].norm(p=1).item()]
                optimizer.step()
                # move any out-of-bound cell back to interposer region
                model.move_boundary_op(pos)
                #print("pos after move:\n", self.pos[0].data, '\n', self.pos[1].data)
                #logging.debug("optimization step %.3f ms" % ((time.time() - t0) * 1000))

                cur_metric.lr = [
                    optimizer.param_groups[0]["lr"].data, 
                    optimizer.param_groups[1]["lr"].data
                    ]
                cur_metric.evaluate(eval_ops, pos)

                # record the best outer cell overflow
                if best_metric[0] is None:
                    best_metric[0] = cur_metric
                    best_pos[0] = [self.pos[0].data.clone(), self.pos[1].data.clone()]
                else:
                    wl_ovlf_flag = cur_metric.overflow < max(
                        best_metric[0].overflow*1.05, self.stop_overflow
                    ) and cur_metric.hpwl < best_metric[0].hpwl
                    temp_flag = cur_metric.temp_max < best_metric[0].temp_max and\
                                cur_metric.hpwl < best_metric[0].hpwl*1.05 if self.temp_aware_opt else False
                    if wl_ovlf_flag or temp_flag:
                        best_metric[0] = cur_metric
                        best_pos[0][0].data.copy_(self.pos[0].data)
                        best_pos[0][1].data.copy_(self.pos[1].data)

                if params.plot_freq >0 and (iteration % params.plot_freq == 1):
                    #logging.info("optimizer step %.3f ms" % ((time.time() - t1) * 1000))
                    cur_pos = [
                        self.pos[0].data.clone().cpu().numpy(), 
                        self.pos[1].data.clone().cpu().numpy()
                    ]
                    plot_fp(system, pos, 1, 1)
                if (iteration % 100 == 1): #iteration % (params.plot_freq//10) == 1:
                    logging.info(cur_metric)
                    #logging.info("full step %.3f ms" % ((time.time() - t0) * 1000))

            iteration = 0
            best_metric = [None]
            best_pos = [None]

            tt = time.time()
            # construct model and optimizer
            loss_weights = [self.density_weight_init]
            if self.temp_aware_opt:
                loss_weights.append(self.temp_weight_init)
                    
            model = PlaceObj(
                loss_weights, params, self.datacollection, floorplan_params
            ).to(self.pos[0].device)
            model.temp_aware_opt = self.temp_aware_opt
            model.temp_weight = 0

            # defining evaluation ops
            eval_ops = {"density": model.density_bell,  "hpwl": model.wirelength_op}
            if self.temp_aware_opt:
                eval_ops.update({"physics_op": model.physics_op})
                model.amb = self.amb
                model.powermap = self.powermap
                model.temp_model = self.temp_model
                model.temp_threshold = params.temp_threshold
            else:
                model.temp_model = None
                
            # determine optimizer
            param_groups = [{'params': self.pos[0], 'lr': 1}, {'params': self.pos[1], 'lr': 1}]
            optimizer_name = floorplan_params["optimizer"]
            if optimizer_name.lower() == "adam":
                optimizer = torch.optim.Adam(param_groups)
            elif optimizer_name.lower() == "sgd":
                optimizer = torch.optim.SGD(param_groups)
            elif optimizer_name.lower() == "sgd_momentum":
                optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=False)
            elif optimizer_name.lower() == "sgd_nesterov":
                optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
            elif optimizer_name.lower() == "cgm":
                optimizer = conjugate_grad_opt.CGOptimizer(param_groups)
            else:
                assert 0, "unknown optimizer %s" % (optimizer_name)

            logging.info("use %s optimizer" % (optimizer_name))

            model.train()
            if iteration == 0:
                if params.plot_freq >0:
                    plot_fp(system, self.pos, 1)
                initialize_learning_rate(system)
                model.initialize_weight(self.pos, self.temp_aware_opt)
            print("Initial LR: (%.2e, %.2e), Weight(density, temp): %.2e, %.2e. " 
                  % (optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], 
                     model.density_weight, model.temp_weight))
            logging.info("%s initialization takes %g seconds" % (optimizer_name, (time.time() - tt)))

            # use a moving average window for stopping criteria, for an example window of 3
            # 0, 1, 2, 3, 4, 5, 6
            #    window2
            #             window1
            self.moving_avg_window = max(min(model.sub_iterations // 2, 4), 1)
            self.stop_overflow = params.stop_overflow
            min_perturb_interval = 100
            Liter_metrics = all_metrics
            Llambda_flat_iteration = 0

            ### preparation for self-adaptive divergence check
            overflow_list = [1]
            divergence_list = []
            last_perturb_iter = -min_perturb_interval

            for Liter_step in trange(model.Liter_iteration):
                Liter_metrics.append([])
                Llambda_metrics = Liter_metrics[-1]
                pertub_perc = 0
                
                for Llambda_weight_step in range(model.Llambda_weight_iteration):
                    Llambda_metrics.append([])
                    Lsub_metrics = Llambda_metrics[-1]
                    
                    for Lsub_step in range(model.sub_iterations):
                            
                        one_descent_step(
                            Liter_step, Llambda_weight_step, Lsub_step, iteration, Lsub_metrics
                        )
                        iteration += 1
                        last_metric = Lsub_metrics[-1]
                        overflow_list.append(last_metric.overflow)
                        divergence_list.append([last_metric.hpwl, last_metric.overflow])
                        
                        if self.check_divergence(
                            divergence_list, best_metric, window=10, threshold=overflow_list[-1]
                        ):
                            pertub_perc = max(0.2, pertub_perc)
                            
                        if self.check_plateau(divergence_list):
                            ## entropy injection when stuck at very high overflow
                            if (overflow_list[-1] > 0.5):
                                logging.info(f"Stuck at very early @ {iteration},"
                                             f" overflow = {overflow_list[-1]:.3f}.")
                                pertub_perc = max(0.6, pertub_perc)
                            else:
                                pertub_perc = max(0.4, pertub_perc)
                            #self.pos[0].data.copy_(best_pos[-1][0].data)
                            #self.pos[1].data.copy_(best_pos[-1][1].data)
                            
                        # stopping criteria
                        if self.Lsub_stop_criterion(Llambda_weight_step, Lsub_step, Lsub_metrics):
                            break
                            
                    #update density weight
                    Llambda_flat_iteration += 1
                    """
                    if Llambda_flat_iteration>1:
                        mu = model.update_weight_op(Llambda_flat_iteration, Llambda_metrics[-1][-1], 
                                    Llambda_metrics[-2][-1] if len(Llambda_metrics) > 1 else Liter_metrics[-2][-1][-1])
                        model.density_weight *= mu
                        #logging.info(f"Update weight by X{mu:.2e}")
                        if self.temp_aware_opt:
                            model.temp_weight *= mu
                    """
                    if pertub_perc > 0 and params.perturb_flag and \
                        (iteration-last_perturb_iter>min_perturb_interval):
                        pertub_size = 0.1
                        model.noise_op(self.pos, pertub_size, pertub_perc)
                        last_perturb_iter = iteration
                        #print("Liter_stop_criterion or Divergence_criterion satisfied!")
                        logging.info(f"\n\t Add perturbation at {Llambda_metrics[-1][-1]}.")

                    if self.Llambda_stop_criterion(Llambda_weight_step, Llambda_metrics):
                        break

                # update learning rate
                if "learning_rate_decay" in floorplan_params:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= floorplan_params["learning_rate_decay"]

                if not (math.isfinite(last_metric.hpwl) and math.isfinite(last_metric.overflow)):
                    logging.error(
                        f"Invalid metric detected at iteration {iteration}: "
                        f"hpwl={last_metric.hpwl}, overflow={last_metric.overflow}. "
                    )
                    break
            logging.info("optimizer %s takes %.3f seconds" % (optimizer_name, time.time() - tt))
            if best_pos[-1] is None:
                raise RuntimeError("No solution found!!!")
            if params.plot_freq > 0:
                plot_fp(system, best_pos[0], 1)
            return best_metric, best_pos, Liter_metrics