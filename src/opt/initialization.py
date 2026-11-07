import time

import numpy as np
import pulp
import gurobipy as gp
from gurobipy import GRB


def calc_clusters(system, theta):
    """
    Collect pins of each chiplet to one pin/ubump clump
    Input:  net information
    Output:
        connect_matrix: A[i,j] represents statistics of connection from i to j
            [:, :, 0] = connection count
            [:, :, 1] = bandwidth weight (2/len(node_id))
            [:, :, 2] = avg x-offset of pins (after rotation)
            [:, :, 3] = avg y-offset of pins (after rotation)
    """
    num_chiplets = system.num_chiplets
    connect_matrix = np.zeros((num_chiplets, num_chiplets, 4))

    for net_id in system.net_id:
        pins = system.net2pin_map[net_id]
        node_id = system.pin2node_map[pins]
        node_id = node_id[node_id < num_chiplets]
        
        for idx1 in range(len(node_id)):
            for idx2 in range(idx1 + 1, len(node_id)):
                i, j = node_id[idx1], node_id[idx2]

                connect_matrix[i, j, 0] += 1
                connect_matrix[i, j, 1] += 2 / len(node_id)
                connect_matrix[i, j, 2] += (
                    system.pin_offset_x[pins[idx1]] * np.cos(theta[i]) -
                    system.pin_offset_y[pins[idx1]] * np.sin(theta[i])
                )
                connect_matrix[i, j, 3] += (
                    system.pin_offset_x[pins[idx1]] * np.sin(theta[i]) +
                    system.pin_offset_y[pins[idx1]] * np.cos(theta[i])
                )

                connect_matrix[j, i, 0] += 1
                connect_matrix[j, i, 1] += 2 / len(node_id)
                connect_matrix[j, i, 2] += (
                    system.pin_offset_x[pins[idx2]] * np.cos(theta[j]) -
                    system.pin_offset_y[pins[idx2]] * np.sin(theta[j])
                )
                connect_matrix[j, i, 3] += (
                    system.pin_offset_x[pins[idx2]] * np.sin(theta[j]) +
                    system.pin_offset_y[pins[idx2]] * np.cos(theta[j])
                )

    nonzeros = connect_matrix[:, :, 0] != 0
    connect_matrix[nonzeros, 2] /= connect_matrix[nonzeros, 0]
    connect_matrix[nonzeros, 3] /= connect_matrix[nonzeros, 0]

    return connect_matrix


def Init_woflip_pulp(system, max_size, overlap_ratio=0.1, TimeLimit=1e3, OutputFlag=1, num_threads=None):
    t1 = time.time()
    # Create the model
    model = pulp.LpProblem("initialization", pulp.LpMinimize)
    
    # PuLP doesn't have direct time limit control, but we can track time manually
    start_time = time.time()
    
    num_chiplets = system.num_chiplets
    num_nodes = system.num_nodes
    size_x, size_y = system.node_size_x[:num_chiplets], system.node_size_y[:num_chiplets]
    xcenter, ycenter = (system.xlow+system.xhigh)/2, (system.ylow+system.yhigh)/2
    connect_matrix = calc_clusters(system, np.zeros(num_chiplets))
    W, H = system.intp_width, system.intp_height
    
    # Create variables
    x = {i: pulp.LpVariable(f"x_{i}", lowBound=system.xlow, upBound=system.xhigh) 
         for i in range(num_chiplets)}
    y = {i: pulp.LpVariable(f"y_{i}", lowBound=system.ylow, upBound=system.yhigh) 
         for i in range(num_chiplets)}
    rx = {i: pulp.LpVariable(f"rx_{i}", cat='Binary') for i in range(num_chiplets)}
    ry = {i: pulp.LpVariable(f"ry_{i}", cat='Binary') for i in range(num_chiplets)}
    
    # Wirelength objective
    wirelength = 0
    abs_vars = {}  # To store absolute value variables
    
    for i in range(num_chiplets):
        for j in range(i+1, num_chiplets):
            if connect_matrix[i,j,0] > 0:
                bandwidth = connect_matrix[i,j,1]
                
                # Distance calculation with rotation
                cosi = 1 - rx[i] - ry[i]
                sini = ry[i] - rx[i]
                cosj = 1 - rx[j] - ry[j]
                sinj = ry[j] - rx[j]
                
                # x distance
                dist_x = pulp.LpVariable(f"distx_{i}_{j}", lowBound=None, upBound=None)
                model += dist_x == (x[i] + connect_matrix[i,j,2]*cosi - connect_matrix[i,j,3]*sini - 
                                   (x[j] + connect_matrix[j,i,2]*cosj - connect_matrix[j,i,3]*sinj))
                
                hpwl_x = pulp.LpVariable(f"hpwl_x_{i}_{j}", lowBound=0)
                model += hpwl_x >= dist_x
                model += hpwl_x >= -dist_x

                # y distance
                dist_y = pulp.LpVariable(f"disty_{i}_{j}", lowBound=None, upBound=None)
                model += dist_y == (y[i] + connect_matrix[i,j,2]*sini + connect_matrix[i,j,3]*cosi - 
                                   (y[j] + connect_matrix[j,i,2]*sinj + connect_matrix[j,i,3]*cosj))

                hpwl_y = pulp.LpVariable(f"hpwl_y_{i}_{j}", lowBound=0)
                model += hpwl_y >= dist_y
                model += hpwl_y >= -dist_y
                
                wirelength += bandwidth * (hpwl_x + hpwl_y)
    
    model += wirelength
    
    # Width and height with rotation
    width = {}
    height = {}
    for i in range(num_chiplets):
        # Width calculation
        width_tmp = pulp.LpVariable(f"widthtmp_{i}", lowBound=None, upBound=None)
        model += width_tmp == (size_x[i]*(1 - rx[i] - ry[i]) + size_y[i]*(ry[i] - rx[i]))
        
        # Absolute value of width
        width[i] = pulp.LpVariable(f"width_{i}", lowBound=0)
        model += width[i] >= width_tmp
        model += width[i] >= -width_tmp
                
        # Height calculation
        height_tmp = pulp.LpVariable(f"heighttmp_{i}", lowBound=None, upBound=None)
        model += height_tmp == (size_x[i]*(ry[i] - rx[i]) + size_y[i]*(1 - rx[i] - ry[i]))
        
        # Absolute value of height        
        height[i] = pulp.LpVariable(f"height_{i}", lowBound=0)
        model += height[i] >= height_tmp
        model += height[i] >= -height_tmp
        
    # Non-overlap constraints
    for i in range(num_chiplets):
        for j in range(i+1, num_chiplets):
            delta1 = pulp.LpVariable(f"delta1_{i}_{j}", cat='Binary')
            delta2 = pulp.LpVariable(f"delta2_{i}_{j}", cat='Binary')
            delta3 = pulp.LpVariable(f"delta3_{i}_{j}", cat='Binary')
            delta4 = pulp.LpVariable(f"delta4_{i}_{j}", cat='Binary')
            
            # Big-M constraints for non-overlap
            M = max(W, H)  # Large enough number
            model += x[i] + (width[i] + width[j]) * overlap_ratio <= x[j] + M * delta1
            model += x[j] + (width[i] + width[j]) * overlap_ratio <= x[i] + M * delta2
            model += y[i] + (height[i] + height[j]) * overlap_ratio <= y[j] + M * delta3
            model += y[j] + (height[i] + height[j]) * overlap_ratio <= y[i] + M * delta4
            model += delta1 + delta2 + delta3 + delta4 <= 3
    
    # Solve the problem
    # Note: PuLP doesn't have built-in time limits for all solvers
    # Using CBC solver which is commonly available
    solver = pulp.PULP_CBC_CMD(msg=OutputFlag, timeLimit=TimeLimit, threads=num_threads)
    model.solve(solver)
    
    print(f"Optimize takes {time.time()-t1:.2e} s")
    
    if model.status == pulp.LpStatusOptimal:
        print("\t!Initial floorplan found.")
    elif time.time() - start_time >= TimeLimit and any(v.varValue is not None for v in model.variables()):
        print("\t!Time limit reached with solution found.")
    else:
        print("\t!Initial solution not found!")
        return None, None, None
    
    # Extract solution
    init_pos = np.zeros((4, num_chiplets))
    for i in range(num_chiplets):
        init_pos[0,i] = x[i].varValue
        init_pos[1,i] = y[i].varValue
        init_pos[2,i] = rx[i].varValue
        init_pos[3,i] = ry[i].varValue
    
    # Center the solution
    init_pos[0] = init_pos[0] - ((init_pos[0]-size_x/2).min()+(init_pos[0]+size_x/2).max())/2 + xcenter
    init_pos[1] = init_pos[1] - ((init_pos[1]-size_y/2).min()+(init_pos[1]+size_y/2).max())/2 + ycenter
    init_angle = np.where((np.abs(init_pos[2,:])<0.1), np.pi*np.sign(init_pos[3,:])/2, 
                         np.pi+np.pi*(1-np.sign(init_pos[3,:]))/2)
    
    return [[init_pos[0], init_pos[1]], init_angle]

def Init_woflip_grb(system, max_size, overlap_ratio=0.1, TimeLimit=1e3, OutputFlag=1, num_threads=None):
    t1 = time.time()
    gpmodel = gp.Model("initialization")
    gpmodel.setParam("OutputFlag", OutputFlag)
    gpmodel.setParam('TimeLimit', TimeLimit)
    if num_threads is not None:
        gpmodel.setParam('Threads', num_threads)

    num_chiplets = system.num_chiplets
    num_nodes = system.num_nodes
    size_x, size_y = system.node_size_x[:num_chiplets], system.node_size_y[:num_chiplets]
    xcenter, ycenter = (system.xlow+system.xhigh)/2, (system.ylow+system.yhigh)/2
    connect_matrix = calc_clusters(system, np.zeros(num_chiplets))
    W, H = system.intp_width, system.intp_height
    
    x, y, rx, ry = {}, {}, {}, {}
    for i in range(num_chiplets):
        x[i] = gpmodel.addVar(lb=system.xlow, ub=system.xhigh, name=f"x_{i}")
        y[i] = gpmodel.addVar(lb=system.ylow, ub=system.yhigh, name=f"y_{i}")
        rx[i] = gpmodel.addVar(vtype=GRB.BINARY, name=f"rx_{i}")
        ry[i] = gpmodel.addVar(vtype=GRB.BINARY, name=f"ry_{i}")
    gpmodel.update()

    wirelength = gp.LinExpr()
    for i in range(num_chiplets):
        for j in range(i+1, num_chiplets):
            if connect_matrix[i,j,0]>0:
                bandwidth = connect_matrix[i,j,1]
                
                #wirelength.add(bandwidth*(dist_x*dist_x+dist_y*dist_y))    
                dist_x = gpmodel.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"distx_{i}")
                cosi, sini = 1-rx[i]-ry[i], ry[i]-rx[i]
                cosj, sinj = 1-rx[j]-ry[j], ry[j]-rx[j]
                gpmodel.addConstr(dist_x == (
                    (x[i] + connect_matrix[i,j,2]*cosi - connect_matrix[i,j,3]*sini) -
                    (x[j] + connect_matrix[j,i,2]*cosj - connect_matrix[j,i,3]*sinj)
                ))
                wlx = gpmodel.addVar(lb=0, ub=GRB.INFINITY, name=f"wlx_{i}")
                gpmodel.addConstr(wlx == gp.abs_(dist_x))
                
                dist_y = gpmodel.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"disty_{i}")
                gpmodel.addConstr(dist_y == (
                    (y[i] + connect_matrix[i,j,2]*sini + connect_matrix[i,j,3]*cosi)-
                    (y[j] + connect_matrix[j,i,2]*sinj + connect_matrix[j,i,3]*cosj)
                ))
                wly = gpmodel.addVar(lb=0, ub=GRB.INFINITY, name=f"wly_{i}")
                gpmodel.addConstr(wly == gp.abs_(dist_y))
                wirelength.add(bandwidth*(wlx+wly))

    gpmodel.setObjective(wirelength, sense=GRB.MINIMIZE)
    
    width, height = {}, {}
    for i in range(num_chiplets):
        width[i] = gpmodel.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"width_{i}")
        height[i] = gpmodel.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"height_{i}")
        
        width_tmp = gpmodel.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"widthtmp_{i}")
        gpmodel.addConstr(width_tmp == (size_x[i]*(1-rx[i]-ry[i])+size_y[i]*(ry[i]-rx[i])))
        gpmodel.addConstr(width[i] == gp.abs_(width_tmp))

        height_tmp = gpmodel.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"heighttmp_{i}")
        gpmodel.addConstr(height_tmp == (size_x[i]*(ry[i]-rx[i])+size_y[i]*(1-rx[i]-ry[i])))
        gpmodel.addConstr(height[i] == gp.abs_(height_tmp))

    for i in range(num_chiplets):
        for j in range(i+1, num_chiplets):
            delta1 = gpmodel.addVar(vtype=GRB.BINARY, name=f"delta1_{i}_{j}")
            delta2 = gpmodel.addVar(vtype=GRB.BINARY, name=f"delta2_{i}_{j}")
            delta3 = gpmodel.addVar(vtype=GRB.BINARY, name=f"delta3_{i}_{j}")
            delta4 = gpmodel.addVar(vtype=GRB.BINARY, name=f"delta4_{i}_{j}")
            
            gpmodel.addConstr(x[i] + (width[i]+width[j])*overlap_ratio <= x[j] + W*delta1)
            gpmodel.addConstr(x[j] + (width[i]+width[j])*overlap_ratio <= x[i] + W*delta2)
            gpmodel.addConstr(y[i] + (height[i]+height[j])*overlap_ratio <= y[j] + H*delta3)
            gpmodel.addConstr(y[j] + (height[i]+height[j])*overlap_ratio <= y[i] + H*delta4)
            gpmodel.addConstr(delta1 + delta2 + delta3 + delta4 <=3)
            ###Note: the effect of size_x&y is almost the same as width and height,
            ###      better than just size for all
            
    gpmodel.update()
    #print(f"Init takes {time.time()-t1:.2e} s, # of variables: {gpmodel.numVars}"+\
    #      f" # of constraints: {gpmodel.numGenConstrs}")

    gpmodel.optimize()
    print(f"Optimize takes {time.time()-t1:.2e} s")

    if gpmodel.SolCount > 0:
        status = gpmodel.Status
        if status == gp.GRB.OPTIMAL:
            print("\t!Optimal initial floorplan found.")
        elif status == gp.GRB.TIME_LIMIT:
            print("\t!Time limit reached, feasible solution found.")
        else:
            print(f"\t!Feasible solution found (status={status}).")
    else:
        print("\t!No feasible initial solution found!")
        gpmodel.dispose()
        return None

    init_pos = np.zeros((4, num_chiplets))
    for i in range(num_chiplets):
        init_pos[0,i] = x[i].x
        init_pos[1,i] = y[i].x
        init_pos[2,i] = rx[i].x
        init_pos[3,i] = ry[i].x
    
    init_pos[0] = init_pos[0] - ((init_pos[0]-size_x/2).min()+(init_pos[0]+size_x/2).max())/2 + xcenter
    init_pos[1] = init_pos[1] - ((init_pos[1]-size_y/2).min()+(init_pos[1]+size_y/2).max())/2 + ycenter
    init_angle = np.where((np.abs(init_pos[2,:])<0.1), np.pi*np.sign(init_pos[3,:])/2, 
                          np.pi+np.pi*(1-np.sign(init_pos[3,:]))/2)
    gpmodel.dispose()
    return [[init_pos[0], init_pos[1]], init_angle]
