import re
import os
import sys
os.environ["OMP_NUM_THREADS"] = "64"

import time
import copy
import inspect #inspect.signature(func).parameters

import torch
torch.set_num_threads(64)
import numpy as np

from utils.visualize_temp import plot_fp

from .nonlinear_place import NonlinearPlace
from .place_obj import PlaceObj
from .initialization import Init_woflip_pulp, Init_woflip_grb
from .legalization import Legalization


def placeflow_core(params, system, compact_model=None):
    """
    Core placement optimization. Returns optimized positions without physical evaluation.
    Returns: (best_pos, best_metric, hpwl, Lgamma_metrics)
    """
    t1 = time.time()
    num_chiplets = system.num_chiplets
    num_nodes = system.num_nodes
    max_intp_size = int(system.area_cplt**0.5) * 0.45
    
    # Chiplet reflection is disabled; initialization optimizes positions and rotations only.
    def find_init(params):
        if params.ilp_solver == 'pulp':
            res = Init_woflip_pulp(system, max_intp_size, params.overflow_init, TimeLimit=params.initialization_time_limit_s, OutputFlag=1, num_threads=params.num_threads)
        elif params.ilp_solver == 'grb':
            res = Init_woflip_grb(system, max_intp_size, params.overflow_init, TimeLimit=params.initialization_time_limit_s, OutputFlag=1, num_threads=params.num_threads)
        return res

    init_res = find_init(params)
    while init_res is None:
        params.overflow_init *= 0.8
        init_res = find_init(params)
    init_pos, init_angle = init_res
    system.node_flip[:num_chiplets] = np.zeros_like(init_angle)

    system.node_x[:num_chiplets] = init_pos[0]
    system.node_y[:num_chiplets] = init_pos[1]
    system.node_orient[:num_chiplets] = init_angle + np.pi * 0.1 * np.random.randn(num_chiplets)
    
    placer = NonlinearPlace(params, system)
    placer.temp_aware_opt = params.temp_aware_opt
    placer.density_weight_init = params.density_weight_init
    placer.temp_weight_init = getattr(params, "temp_weight_init", 0)

    if params.temp_aware_opt and compact_model:
        placer.temp_model = compact_model.get('Thermal', None)
        placer.powermap = torch.Tensor(system.powermap).reshape(1, -1, 1)

    model = PlaceObj([0, 0], params, placer.datacollection, params.floorplan_stages[0])
    placer.amb = model.amb = params.ambient

    try:

        best_metric, best_pos, Lgamma_metrics = placer(params, system)
        _, theta, size_x, size_y = model.legalize_theta(best_pos[0][1].data.clone())
        size_x, size_y = size_x.cpu().numpy(), size_y.cpu().numpy()
        # Legalization must rotate pin offsets using the final snapped orientations.
        legal_system = copy.copy(system)
        legal_system.node_orient = theta.detach().cpu().numpy()
        x_init = best_pos[0][0][:num_chiplets].detach().cpu().numpy()
        y_init = best_pos[0][0][num_nodes:num_nodes + num_chiplets].detach().cpu().numpy()

        print(f"Legalization with WL weight {params.wl_weight:.3e}!")
        final_pos = Legalization(
            legal_system, params.ilp_solver, params.wl_weight, [[x_init, y_init]], 
            [size_x+params.dis_bet_chips, size_y+params.dis_bet_chips], 
            TimeLimit=params.legalization_time_limit_s, num_threads=params.num_threads
        )

        if final_pos is None:
            print("WL legalization returned no result; retrying with displacement only.")
            final_pos = Legalization(
                legal_system, params.ilp_solver, 0, [[x_init, y_init]],
                [size_x+params.dis_bet_chips, size_y+params.dis_bet_chips],
                TimeLimit=params.legalization_time_limit_s, num_threads=params.num_threads
            )
            
        if final_pos is None:
            print("No feasible floorplan!")
            raise ValueError("No final feasible floorplan found!")
    
    except Exception as e:
        raise ValueError(f"Error {e} happen during placeflow!")
        import traceback
        traceback.print_exc()
    
    print(f"Total time {time.time()-t1:.3f}")
    best_pos[0][0][:num_chiplets] = torch.from_numpy(final_pos[0])
    best_pos[0][0][num_nodes:num_nodes+num_chiplets] = torch.from_numpy(final_pos[1])
    best_pos[0][1] = theta
    if params.plot_freq > 0:
        plot_fp(system, best_pos[0], 1)
    with torch.no_grad():
        hpwl = model.hpwl_op(best_pos[0]).item()

    return {
        "hpwl": hpwl,
        "best_fp_pos": best_pos, "best_fp_size": [size_x, size_y], 
        "best_metric": best_metric, "Lgamma_metrics": Lgamma_metrics, 
        }
