"""Command-line selectors; training and simulation settings come from JSON."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from .training import generate_dataset, fit_model
from .atsim import evaluate_layout


TRAINING_FIELDS = {
    'train_samples': 'training_samples',
    'test_samples': 'training_test_samples',
    'iterations': 'training_iterations',
    'learning_rate': 'training_learning_rate',
    'seed': 'training_seed',
    'lr_decay_step': 'training_lr_decay_step',
    'lr_decay_gamma': 'training_lr_decay_gamma',
    'dataset': 'training_dataset',
    'result_dir': 'training_result_dir',
    'ilp_solver': 'training_ilp_solver',
    'sampling_time_limit_s': 'training_sampling_time_limit_s',
}


def configured_path(value, root, case, timestamp):
    path = Path(value.format(case=case, timestamp=timestamp)).expanduser()
    return path if path.is_absolute() else root / path


def validate_training_settings(settings):
    for key, minimum in [('train_samples',1),('test_samples',0),('iterations',1),('lr_decay_step',1),('seed',0)]:
        value = settings[key]
        if not isinstance(value,int) or isinstance(value,bool) or value < minimum:
            raise ValueError(f'thermal.{TRAINING_FIELDS[key]} must be an integer >= {minimum}')
    for key in ['learning_rate','lr_decay_gamma','sampling_time_limit_s']:
        value = settings[key]
        if not isinstance(value,(float,int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'thermal.{TRAINING_FIELDS[key]} must be finite and positive')
    if settings['ilp_solver'] not in (None,'pulp','grb'):
        raise ValueError('thermal.training_ilp_solver must be null, pulp or grb')


def train_main():
    from ..opt import cli as placement
    parser = argparse.ArgumentParser(description="Train a thermal model using the flat thermal section in parameters")
    parser.add_argument('--case', required=True)
    parser.add_argument('--param-file', '--param_file', dest='param_file', help="Case JSON overrides, default: cases/CASE/Thermal-aware.json")
    args = parser.parse_args()
    case_dir = placement.default_case_dir(args.case)
    param_file = Path(args.param_file).resolve() if args.param_file else case_dir/'Thermal-aware.json'
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    params = placement.load_params(param_file, args.case, placement.ROOT)
    settings = {k: getattr(params, v) for k, v in TRAINING_FIELDS.items()}
    validate_training_settings(settings)
    out = configured_path(settings['result_dir'], placement.ROOT, args.case, timestamp)
    out.mkdir(parents=True, exist_ok=True)
    params.result_dir = str(out)
    if settings['ilp_solver'] is not None:
        params.ilp_solver = settings['ilp_solver']
    torch.set_num_threads(params.num_threads)
    system = placement.build_system(Path(params.case_dir), args.case, params)
    if settings['dataset']:
        dataset_path = configured_path(settings['dataset'], placement.ROOT, args.case, timestamp)
        saved = torch.load(dataset_path, map_location='cpu', weights_only=True)
    else:
        saved = generate_dataset(params, system, out)
    report = fit_model(saved, params, system, params.thermal_model, out)
    print(json.dumps(report, indent=2))
    return 0


def evaluate_main():
    from ..opt import cli as placement
    parser = argparse.ArgumentParser(description="Evaluate a layout using thermal settings in parameters")
    parser.add_argument('--layout', required=True, help="layout.json written by run_placement_opt.py")
    parser.add_argument('--param-file', '--param_file', dest='param_file', help="Case JSON overrides, default: the case Thermal-aware.json")
    args = parser.parse_args()
    layout = placement.load_json(Path(args.layout).resolve())
    case = layout['case']
    param_file = Path(args.param_file).resolve() if args.param_file else placement.default_case_dir(case)/'Thermal-aware.json'
    params = placement.load_params(param_file, case, placement.ROOT)
    out = configured_path(params.simulation_result_dir, placement.ROOT, case, time.strftime('%Y%m%d_%H%M%S'))
    result = evaluate_layout(layout, params, out)
    print(json.dumps(result, indent=2))
    return 0
