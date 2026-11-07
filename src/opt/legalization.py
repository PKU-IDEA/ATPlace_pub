import pulp
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .initialization import calc_clusters


def Legalization(system, solver, wl_weight, init_poses, init_sizes, TimeLimit=1e2, OutputFlag=0, num_threads=None, random_seed=None):
    if solver.lower()=='pulp':
        return Legalization_pulp(
            system, wl_weight, init_poses, init_sizes,
            TimeLimit=TimeLimit, OutputFlag=OutputFlag, num_threads=num_threads, random_seed=random_seed
        )
    elif solver.lower() in ('grb','gurobi'):
        return Legalization_grb(
            system, wl_weight, init_poses, init_sizes,
            TimeLimit=TimeLimit, OutputFlag=OutputFlag, num_threads=num_threads, random_seed=random_seed
        )
    else:
        raise ValueError(f"Only <pulp>/<grb> supported for ILP, rather than {solver}")
        
def Legalization_pulp(system, wl_weight, legal_pos, legal_size, TimeLimit=1e2, OutputFlag=0, num_threads=None, random_seed=None):
    num_chiplets = system.num_chiplets
    num_nodes = system.num_nodes
    cos_legal = np.cos(system.node_orient).reshape(-1, 1)
    sin_legal = np.sin(system.node_orient).reshape(-1, 1)

    x_init, y_init = legal_pos[0]
    size_x, size_y = legal_size
    W, H = system.intp_width, system.intp_height

    # Create problem
    prob = pulp.LpProblem("legalization", pulp.LpMinimize)

    x = {}
    y = {}
    for i in range(num_chiplets):
        x[i] = pulp.LpVariable(f"x_{i}", 
                               lowBound=system.xlow + size_x[i]/2,
                               upBound=system.xhigh - size_x[i]/2)
        y[i] = pulp.LpVariable(f"y_{i}",
                               lowBound=system.ylow + size_y[i]/2,
                               upBound=system.yhigh - size_y[i]/2)

    objective = 0
    objname = " Dis "
    displacex = {}
    displacey = {}

    for i in range(num_chiplets):
        disp_x = pulp.LpVariable(f"disp_x{i}", cat='Continuous')
        prob += disp_x == x[i] - x_init[i]

        displacex[i] = pulp.LpVariable(f"displacex_{i}", lowBound=0)
        prob += displacex[i] >= disp_x
        prob += displacex[i] >= -disp_x

        disp_y = pulp.LpVariable(f"disp_y{i}", cat='Continuous')
        prob += disp_y == y[i] - y_init[i]

        displacey[i] = pulp.LpVariable(f"displacey_{i}", lowBound=0)
        prob += displacey[i] >= disp_y
        prob += displacey[i] >= -disp_y

        objective += displacex[i] + displacey[i]

    if wl_weight > 0:
        objname += " WL "
        for net_id in range(len(system.net2pin_map)):
            pins = system.net2pin_map[net_id]
            nodes = system.pin2node_map[pins]
            idx1, idx2 = nodes[0], nodes[1]
            pin1, pin2 = pins[0], pins[1]

            # X direction
            dist_x = pulp.LpVariable(f"distx_{net_id}", cat='Continuous')
            term1 = (system.pin_offset_x[pin1] * cos_legal[idx1] -
                     system.pin_offset_y[pin1] * sin_legal[idx1])
            term2 = (system.pin_offset_x[pin2] * cos_legal[idx2] -
                     system.pin_offset_y[pin2] * sin_legal[idx2])
            prob += dist_x == x[idx1] - x[idx2] + term1 - term2

            hpwl_x = pulp.LpVariable(f"hpwl_x_{net_id}", lowBound=0)
            prob += hpwl_x >= dist_x
            prob += hpwl_x >= -dist_x

            # Y direction
            dist_y = pulp.LpVariable(f"disty_{net_id}", cat='Continuous')
            term3 = (system.pin_offset_x[pin1] * sin_legal[idx1] +
                     system.pin_offset_y[pin1] * cos_legal[idx1])
            term4 = (system.pin_offset_x[pin2] * sin_legal[idx2] +
                     system.pin_offset_y[pin2] * cos_legal[idx2])
            prob += dist_y == y[idx1] - y[idx2] + term3 - term4

            hpwl_y = pulp.LpVariable(f"hpwl_y_{net_id}", lowBound=0)
            prob += hpwl_y >= dist_y
            prob += hpwl_y >= -dist_y

            objective += wl_weight * (hpwl_x + hpwl_y)

    prob += objective

    # Non-overlap constraints
    for i in range(num_chiplets):
        for j in range(i+1, num_chiplets):
            delta1 = pulp.LpVariable(f"delta1_{i}_{j}", cat=pulp.LpBinary)
            delta2 = pulp.LpVariable(f"delta2_{i}_{j}", cat=pulp.LpBinary)
            delta3 = pulp.LpVariable(f"delta3_{i}_{j}", cat=pulp.LpBinary)
            delta4 = pulp.LpVariable(f"delta4_{i}_{j}", cat=pulp.LpBinary)

            prob += x[i] + (size_x[i]/2 + size_x[j]/2) <= x[j] + W * delta1
            prob += x[j] + (size_x[i]/2 + size_x[j]/2) <= x[i] + W * delta2
            prob += y[i] + (size_y[i]/2 + size_y[j]/2) <= y[j] + H * delta3
            prob += y[j] + (size_y[i]/2 + size_y[j]/2) <= y[i] + H * delta4
            prob += delta1 + delta2 + delta3 + delta4 <= 3

    # Solve with PuLP solver
    # CBC's CLP randomSeed=0 selects a clock-based seed; use 1 for configured zero.
    options = [] if random_seed is None else [f'randomSeed {random_seed or 1}', f'randomCbcSeed {random_seed or 1}']
    solver = pulp.PULP_CBC_CMD(msg=OutputFlag, timeLimit=TimeLimit, threads=num_threads, options=options)
    prob.solve(solver)
    
    new_pos = np.zeros((2, num_chiplets))
    if pulp.LpStatus[prob.status] == 'Optimal':
        print(f"!Legalized Floorplan of {objname} found!")
        for i in range(num_chiplets):
            new_pos[0,i] = pulp.value(x[i])
            new_pos[1,i] = pulp.value(y[i])
            if OutputFlag:
                print(f"\tOffset {i}: x={pulp.value(x[i])-x_init[i]}, y={pulp.value(y[i])-y_init[i]}")
    else:
        print(f"!Legalized Floorplan of {objname} not found!")
        new_pos = None
    
    return new_pos

def Legalization_grb(system, wl_weight, legal_pos, legal_size, TimeLimit=1e2, OutputFlag=0, num_threads=None, random_seed=None):
    
    legalizer = gp.Model("legalization")
    legalizer.setParam("OutputFlag", OutputFlag)
    legalizer.setParam('TimeLimit', TimeLimit)
    if num_threads is not None:
        legalizer.setParam('Threads', num_threads)
    if random_seed is not None:
        legalizer.setParam('Seed', random_seed)

    num_chiplets = system.num_chiplets
    num_nodes = system.num_nodes
    x_init, y_init = legal_pos[0]
    size_x, size_y = legal_size
    W, H = system.intp_width, system.intp_height
    
    x, y = {}, {}
    for i in range(num_chiplets):
        x[i] = legalizer.addVar(lb=system.xlow+size_x[i]/2, 
                                ub=system.xhigh-size_x[i]/2, name=f"x_{i}")
        y[i] = legalizer.addVar(lb=system.ylow+size_y[i]/2, 
                                ub=system.yhigh-size_y[i]/2, name=f"y_{i}")
    legalizer.update()
    
    objective = gp.LinExpr()
    for i in range(num_chiplets):
        disp_x = legalizer.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"disp_x{i}")
        legalizer.addConstr(disp_x == x[i]-x_init[i])                
        displacex = legalizer.addVar(lb=0, ub=GRB.INFINITY, name=f"displacex_{i}")
        legalizer.addConstr(displacex == gp.abs_(disp_x))
        disp_y = legalizer.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"disp_y{i}")
        legalizer.addConstr(disp_y == y[i]-y_init[i])                
        displacey = legalizer.addVar(lb=0, ub=GRB.INFINITY, name=f"displacey_{i}")
        legalizer.addConstr(displacey == gp.abs_(disp_y))
        objective.add((displacex+displacey))

    if wl_weight>0:
        cos_legal = np.cos(system.node_orient).reshape(-1,1)
        sin_legal = np.sin(system.node_orient).reshape(-1,1)
        for net_id in range(len(system.net2pin_map)):
            pins = system.net2pin_map[net_id]
            nodes = system.pin2node_map[pins]
            idx1, idx2 = nodes[0], nodes[1]
            pin1, pin2 = pins[0], pins[1]
            dist_x = legalizer.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"distx_{i}")
            legalizer.addConstr(dist_x == x[idx1]-x[idx2]+\
                                (system.pin_offset_x[pin1]*cos_legal[idx1]-\
                                system.pin_offset_y[pin1]*sin_legal[idx1])-\
                                (system.pin_offset_x[pin2]*cos_legal[idx2]-\
                                system.pin_offset_y[pin2]*sin_legal[idx2]))
            hpwl_x = legalizer.addVar(lb=0, ub=GRB.INFINITY, name=f"hpwl_x_{i}")
            legalizer.addConstr(hpwl_x == gp.abs_(dist_x))

            dist_y = legalizer.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"disty_{i}")
            legalizer.addConstr(dist_y == y[idx1]-y[idx2]+\
                                (system.pin_offset_x[pin1]*sin_legal[idx1]+\
                                system.pin_offset_y[pin1]*cos_legal[idx1])-\
                                (system.pin_offset_x[pin2]*sin_legal[idx2]+\
                                system.pin_offset_y[pin2]*cos_legal[idx2]))
            hpwl_y = legalizer.addVar(lb=0, ub=GRB.INFINITY, name=f"hpwl_y_{i}")
            legalizer.addConstr(hpwl_y == gp.abs_(dist_y))
            objective.add(wl_weight*(hpwl_x+hpwl_y))   

    elif wl_weight<0:
        connect_matrix = calc_clusters(system, system.node_orient)
        for i in range(num_chiplets):
            for j in range(i+1, num_chiplets):
                if connect_matrix[i,j,0]>0:
                    bandwidth = connect_matrix[i,j,1]

                    dist_x = legalizer.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"distx_{i}")
                    legalizer.addConstr(dist_x == (
                        (x[i] + connect_matrix[i,j,2]) - (x[j] + connect_matrix[j,i,3])
                    ))
                    wlx = legalizer.addVar(lb=0, ub=GRB.INFINITY, name=f"wlx_{i}")
                    legalizer.addConstr(wlx == gp.abs_(dist_x))

                    dist_y = legalizer.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"disty_{i}")
                    legalizer.addConstr(dist_y == (
                        (y[i] + connect_matrix[i,j,2])- (y[j] + connect_matrix[j,i,3])
                    ))
                    wly = legalizer.addVar(lb=0, ub=GRB.INFINITY, name=f"wly_{i}")
                    legalizer.addConstr(wly == gp.abs_(dist_y))
                    objective.add(bandwidth*(wlx+wly))

    legalizer.update()
    legalizer.setObjective(objective, sense=GRB.MINIMIZE)

    for i in range(num_chiplets):
        for j in range(i+1, num_chiplets):
            delta1 = legalizer.addVar(vtype=GRB.BINARY, name=f"delta1_{i}_{j}")
            delta2 = legalizer.addVar(vtype=GRB.BINARY, name=f"delta2_{i}_{j}")
            delta3 = legalizer.addVar(vtype=GRB.BINARY, name=f"delta3_{i}_{j}")
            delta4 = legalizer.addVar(vtype=GRB.BINARY, name=f"delta4_{i}_{j}")
            
            legalizer.addConstr(x[i] + (size_x[i]/2+size_x[j]/2) <= x[j] + W*delta1)
            legalizer.addConstr(x[j] + (size_x[i]/2+size_x[j]/2) <= x[i] + W*delta2)
            legalizer.addConstr(y[i] + (size_y[i]/2+size_y[j]/2) <= y[j] + H*delta3)
            legalizer.addConstr(y[j] + (size_y[i]/2+size_y[j]/2) <= y[i] + H*delta4)
            legalizer.addConstr(delta1 + delta2 + delta3 + delta4 <=3)
    legalizer.optimize()
    new_pos = np.zeros((2, num_chiplets))
    if legalizer.SolCount > 0:
        for i in range(num_chiplets):
            new_pos[0, i] = x[i].x
            new_pos[1, i] = y[i].x
            if OutputFlag:
                print(f"\tBlock offset {i}: x={x[i].x - x_init[i]}, y={y[i].x - y_init[i]}")
    else:
        if OutputFlag:
            print("No feasible solution found by Gurobi.")
        new_pos = None

    legalizer.dispose()
    return new_pos
