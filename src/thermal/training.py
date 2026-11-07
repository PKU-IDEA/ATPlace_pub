"""Generate ATSim labels and fit the compact temperature-rise model."""
from pathlib import Path
import json
import time
import numpy as np
import pulp
import torch
from .checkpoint import metadata, check_metadata, new_model
from .atsim import ATSimSolver


def sample_layout(system, params, rng):
    """Random quarter-turn orientations and legal positions, without mutating system."""
    n = system.num_chiplets
    rotate = rng.integers(0, 4, n) % 2
    w = np.where(rotate, system.node_size_y[:n], system.node_size_x[:n])
    h = np.where(rotate, system.node_size_x[:n], system.node_size_y[:n])
    margin = params.dis_bet_chips/2
    lo_x, hi_x = system.xlow+w/2+margin, system.xhigh-w/2-margin
    lo_y, hi_y = system.ylow+h/2+margin, system.yhigh-h/2-margin
    if np.any(lo_x > hi_x) or np.any(lo_y > hi_y):
        raise ValueError("Chiplet does not fit inside the placement boundary")
    desired_x, desired_y = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
    problem = pulp.LpProblem("thermal_training_layout", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x{i}", float(lo_x[i]), float(hi_x[i])) for i in range(n)]
    y = [pulp.LpVariable(f"y{i}", float(lo_y[i]), float(hi_y[i])) for i in range(n)]
    objective = []
    for i in range(n):
        for axis, target, label in [(x, desired_x, 'x'), (y, desired_y, 'y')]:
            d = pulp.LpVariable(f"d{label}{i}", lowBound=0)
            problem += d >= axis[i]-float(target[i])
            problem += d >= float(target[i])-axis[i]
            objective.append(d)
        for j in range(i):
            flags = [pulp.LpVariable(f"b{i}_{j}_{k}", cat="Binary") for k in range(4)]
            sx, sy = float((w[i]+w[j])/2+params.dis_bet_chips), float((h[i]+h[j])/2+params.dis_bet_chips)
            problem += x[i]+sx <= x[j]+system.intp_width*flags[0]
            problem += x[j]+sx <= x[i]+system.intp_width*flags[1]
            problem += y[i]+sy <= y[j]+system.intp_height*flags[2]
            problem += y[j]+sy <= y[i]+system.intp_height*flags[3]
            problem += pulp.lpSum(flags) <= 3
    problem += pulp.lpSum(objective)
    if params.ilp_solver == 'pulp':
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=params.training_sampling_time_limit_s, threads=params.num_threads)
    elif params.ilp_solver == 'grb':
        solver = pulp.GUROBI(msg=True, timeLimit=params.training_sampling_time_limit_s, Threads=params.num_threads)
    else:
        raise ValueError("ilp_solver must be grb or pulp")
    problem.solve(solver)
    if problem.sol_status not in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible):
        raise RuntimeError(f"Could not generate a legal thermal training layout within {params.training_sampling_time_limit_s} seconds")
    geometry = np.array([[pulp.value(x[i]), pulp.value(y[i]), w[i], h[i]] for i in range(n)], dtype=np.float32)
    if not np.isfinite(geometry).all():
        raise ValueError("Non-finite sampled layout")
    # Check actual rounded coordinates before constructing a physical simulation.
    for i in range(n):
        for j in range(i):
            if not (abs(geometry[i,0]-geometry[j,0])+.02 >= (w[i]+w[j])/2+params.dis_bet_chips or
                    abs(geometry[i,1]-geometry[j,1])+.02 >= (h[i]+h[j])/2+params.dis_bet_chips):
                raise ValueError("Sampled thermal layout violates chiplet clearance")
    return geometry


def generate_dataset(params, system, out_dir):
    train_samples, test_samples, seed = params.training_samples, params.training_test_samples, params.training_seed
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    geometries, labels = [], []
    signature = metadata(params, system)
    for index in range(train_samples+test_samples):
        geometry = sample_layout(system, params, rng)
        solver = ATSimSolver(out_dir/f"sample_{index:03d}", params)
        solver.set_params(system)
        solver.set_pos(system.powermap, [geometry[:,0], geometry[:,1]], geometry[:,2], geometry[:,3])
        temperature = solver.run()
        geometries.append(torch.from_numpy(geometry))
        labels.append(torch.from_numpy((temperature-params.ambient).copy()).float().unsqueeze(0))
        print(f"ATSim sample {index+1}/{train_samples+test_samples}: Tmax={temperature.max():.2f} C", flush=True)
    saved = {"metadata": signature, "geometry": torch.stack(geometries),
        "power": torch.tensor(system.powermap, dtype=torch.float32).repeat(len(geometries),1),
        "temperature_rise": torch.stack(labels), "train_samples": train_samples,
        "test_samples": test_samples, "seed": seed}
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(saved, out_dir/"dataset.pt")
    return saved


def validate_dataset(saved, params, system):
    check_metadata(saved["metadata"], metadata(params, system))
    count = saved['train_samples']+saved['test_samples']
    n = system.num_chiplets
    expected = {'geometry': (count,n,4), 'power': (count,n),
                'temperature_rise': (count,1,params.num_grid_x,params.num_grid_y)}
    if saved['train_samples'] < 1 or saved['test_samples'] < 0:
        raise ValueError("Invalid train/test split")
    for key, shape in expected.items():
        value = saved[key]
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid dataset tensor: {key}")


def fit_model(saved, params, system, model_path, out_dir):
    iterations, lr, seed = params.training_iterations, params.training_learning_rate, params.training_seed
    validate_dataset(saved, params, system)
    torch.set_num_threads(params.num_threads)
    torch.manual_seed(seed)
    model = new_model(params, system)
    device = torch.device('cpu')
    model.to(device)
    geo = saved['geometry'].to(device)
    power = saved['power'].to(device)
    target = saved['temperature_rise'].to(device)
    inputs = tuple(geo[:,:,i] for i in range(4))+(power,)
    train_count = saved['train_samples']
    train_inputs = tuple(x[:train_count] for x in inputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.training_lr_decay_step, gamma=params.training_lr_decay_gamma)
    with torch.no_grad():
        initial_mse = (model(train_inputs)-target[:train_count]).square().mean().item()
    start = time.time()
    for step in range(iterations):
        optimizer.zero_grad()
        loss = (model(train_inputs)-target[:train_count]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite thermal training loss")
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            model.amp.clamp_(min=1e-6)
            model.heff.clamp_(min=1e-6)
            model.decay.clamp_(min=1e-6)
        if step == 0 or (step+1) % 100 == 0 or step+1 == iterations:
            print(f"Training {step+1}/{iterations}: MSE={loss.item():.6g}", flush=True)
    with torch.no_grad():
        predicted = model(inputs)
        error = predicted-target
        report = {"initial_train_mse": initial_mse,
            "final_train_mse": error[:train_count].square().mean().item(),
            "train_mae_c": error[:train_count].abs().mean().item(),
            "test_mae_c": error[train_count:].abs().mean().item() if saved['test_samples'] else None,
            "train_samples": train_count, "test_samples": saved['test_samples'],
            "iterations": iterations, "learning_rate": lr, "seed": seed,
            "settings": {k:v for k,v in vars(params).items() if k.startswith("training_")},
            "runtime_s": time.time()-start, "metadata": saved['metadata']}
    if not torch.isfinite(predicted).all():
        raise ValueError("Non-finite fitted model predictions")
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": saved['metadata'], "state_dict": model.cpu().state_dict(),
                "training": report}, model_path)
    report['checkpoint'] = str(model_path.resolve())
    (Path(out_dir)/'training_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
