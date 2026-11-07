import time

import torch


class EvalMetrics (object):
    """
    @brief evaluation metrics at one step
    """
    def __init__(self, model, iteration=None, detailed_step=None):
        """
        @brief initialization
        @param iteration optimization step
        """
        self.iteration = iteration
        self.detailed_step = detailed_step
        self.bin_size = model.data_collections.bin_size_x*model.data_collections.bin_size_y
        self.target_dens = model.data_collections.target_dens
        self.total_chip_area = model.data_collections.tot_mov_areas
        self.density_weight = model.density_weight.data
                
        self.objective = None
        self.gradnorm = None
        self.wirelength = None
        self.density_penalty = None
        self.overflow = None
        self.lr = None
        self.hpwl = None
        self.eval_time = None
        self.temp_weight = None
        self.temp_max = -1
        self.temp_grad = None
        self.temp_penalty = None
        self.temp_threshold = None

    def __str__(self):
        """
        @brief convert to string
        """
        content = ""
        if self.iteration is not None:
            content = "Iteration %4d" % (self.iteration)
        if self.detailed_step is not None:
            content += ", (%4d, %2d, %2d)" % (self.detailed_step[0], self.detailed_step[1], self.detailed_step[2])
        if self.objective is not None:
            content += ", Obj %.3E" % (self.objective)
        if self.eval_time is not None:
            content += ", time %.1fms" % (self.eval_time*1000)
        if self.lr is not None:
            content += ", LR (%.2E, %.2E)" % (self.lr[0], self.lr[1])
        if self.gradnorm is not None:
            content += ", Gradnorm (%.1E, %.1E)" % (self.gradnorm[0], self.gradnorm[1])
        if self.wirelength is not None:
            content += ", WL %.3E" % (self.wirelength)
        if self.hpwl is not None:
            content += ", HPWL %.3E" % (self.hpwl)
        if self.overflow is not None:
            content += ", Overflow %.3F" % (self.overflow)
        if self.density_weight is not None:
            content += ", Density_weight %.1E" % (self.density_weight)
        if self.density_penalty is not None:
            content += ", Density_penalty %.2E" % (self.density_penalty)
        if self.temp_weight is not None:
            content += ", Temp_weight %.1E" % (self.temp_weight)
        if self.temp_max is not None:
            content += ", Maximum Temp %.1F" % (self.temp_max)
        if self.temp_grad is not None:
            content += ", Temp Grad %.1F" % (self.temp_grad)
        if self.temp_penalty is not None:
            content += ", Temp Penalty %.1E" % (self.temp_penalty)

        return content

    def __repr__(self):
        """
        @brief print
        """
        return self.__str__()

    def evaluate(self, ops, var):
        """
        @brief evaluate metrics
        @param placedb placement database
        @param ops a list of ops
        @param var variables
        """
        tt = time.time()
        with torch.no_grad():
            if "wirelength" in ops:
                self.wirelength = ops["wirelength"](var).data.item()
            
            if "hpwl" in ops:
                self.hpwl = ops["hpwl"](var).data.item()
            
            if "density" in ops:
                density_map = ops["density"](var)
                self.density_penalty = ((density_map-self.target_dens*self.bin_size)**2).sum().data.item()
                self.overflow = ((density_map.data-self.bin_size).relu_().sum()/self.total_chip_area).item()
            
            if "physics_op" in ops:
                physical_field = ops["physics_op"](var)
                self.temp_map = physical_field['Thermal'].cpu().data
                self.temp_max = self.temp_map.max().item()
                self.temp_grad = (self.temp_map.max()-self.temp_map.min()).item()
                self.temp_penalty = torch.relu(self.temp_map-self.temp_threshold).pow(2).sum().item()
                
                
        self.eval_time = time.time() - tt
