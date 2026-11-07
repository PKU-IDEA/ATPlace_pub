from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
from src import Params
from src.thermal.atsim import temperature_target
from src.opt.chiplet import Chiplet
from src.opt.interposer import Passive_Interposer
from src.opt.system import System_25D
from src.opt.place_flow import placeflow_core
from utils.blocks_parser import parse_blocks
from utils.nets_parser import parse_nets
from utils.pl_parser import parse_pls


def build_compact_model(params: Params.Params, system: System_25D):
    if not params.temp_aware_opt:
        return None
    from src.thermal.checkpoint import load_model
    return {"Thermal": load_model(params.thermal_model, params, system)}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def flatten_numbers(value):
    if isinstance(value, (int, float, np.number)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(flatten_numbers(item))
        return values
    return []


def export_layout(best_fp, system: System_25D, params: Params.Params, case_name: str, mode: str) -> dict:
    raw = to_plain(best_fp)
    pos = raw
    if isinstance(raw, dict):
        for key in ("pos", "position", "best_fp", "best_fp_pos"):
            if key in raw:
                pos = raw[key]
                break

    chiplets = []
    if isinstance(pos, list) and len(pos) >= 2:
        xy = flatten_numbers(pos[0])
        angles = flatten_numbers(pos[1])
        if len(xy) >= 2 * system.num_nodes:
            for idx in range(system.num_chiplets):
                angle = angles[idx] if idx < len(angles) else 0.0
                width = float(system.node_size_x[idx])
                height = float(system.node_size_y[idx])
                # The kernel returns orientations snapped to quarter turns.
                # Export dimensions in the final layout's coordinate axes.
                if int(round(angle / (np.pi / 2))) % 2:
                    width, height = height, width
                chiplets.append({
                    "name": system.node_names[idx],
                    "x": xy[idx],
                    "y": xy[idx + system.num_nodes],
                    "width": width,
                    "height": height,
                    "angle_rad": angle,
                    "power_w": float(system.powermap[idx]) if idx < len(system.powermap) else 0.0,
                })

    return {
        "case": case_name,
        "mode": mode,
        "unit": "um",
        "interposer": {
            "width": float(system.intp_width),
            "height": float(system.intp_height),
            "fence": [float(system.xlow), float(system.xhigh), float(system.ylow), float(system.yhigh)],
        },
        "thermal": {
            "temperature_target": temperature_target(Path(params.thermal_solver).parent / "structure.xml.template"),
            "temp_aware_opt": bool(params.temp_aware_opt),
            "thermal_solver": str(params.thermal_solver),
            "thermal_convection_resistance": float(params.thermal_convection_resistance),
            "ambient_c": float(params.ambient),
            "num_grid_x": int(params.num_grid_x),
            "num_grid_y": int(params.num_grid_y),
        },
        "chiplets": chiplets,
        "raw_best_fp": raw,
    }


def normalize_stage(params: Params.Params, data: dict) -> None:
    keys = ("iteration", "learning_rate", "learning_rate_decay", "optimizer",
            "density_weight_iterations", "sub_iterations")
    default_stage = {key: getattr(params, key) for key in keys}
    # Multiple stages are not fully supported by the placement loop.
    params.floorplan_stages = [default_stage]


def load_params(param_file: Path, case_name: str, out_dir: Path = None, mode: str = None) -> Params.Params:
    if mode not in (None, "wl", "thermal"):
        raise ValueError(f"Unsupported optimization mode: {mode}")
    params = Params.Params(ROOT / "params.json")
    data = load_json(param_file)
    params.fromJson(data)
    # Runtime flag only: configuration files cannot select the optimization mode.
    params.temp_aware_opt = mode == "thermal"
    normalize_stage(params, data)
    params.fence_width = getattr(params, "fence_width", 0.0)
    params.fence_height = getattr(params, "fence_height", 0.0)
    if out_dir is None:
        result_path = Path(params.result_dir.format(
            case=case_name, mode=mode or "wl",
            timestamp=time.strftime("%Y%m%d_%H%M%S"))).expanduser()
        out_dir = result_path if result_path.is_absolute() else ROOT / result_path
    params.result_dir = str(out_dir)
    case_path = Path(params.case_dir.format(case=case_name)).expanduser()
    params.case_dir = str(case_path if case_path.is_absolute() else ROOT / case_path)
    params.interposer_size = load_interposer_size(Path(params.case_dir), case_name)
    solver_path = Path(params.thermal_solver).expanduser()
    params.thermal_solver = str(solver_path if solver_path.is_absolute() else ROOT / solver_path)
    model_path = Path(params.thermal_model.format(case=case_name)).expanduser()
    params.thermal_model = str(model_path if model_path.is_absolute() else ROOT / model_path)
    params.ilp_solver = getattr(params, "ilp_solver", "grb")
    return params


def load_interposer_size(case_dir: Path, case_name: str) -> list:
    path = case_dir / f"{case_name}.blocks"
    _, parsed_headers = parse_blocks({"filename_blocks": str(path)})
    headers = parsed_headers["Headers"]
    dimensions = [headers.get("InterposerWidth"), headers.get("InterposerHeight")]
    if any(value is None or not np.isfinite(value) or value <= 0 for value in dimensions):
        raise ValueError(f"{path}: InterposerWidth and InterposerHeight must be finite positive numbers in micrometers")
    return dimensions


def default_case_dir(case_name: str) -> Path:
    defaults = Params.Params(ROOT / "params.json")
    path = Path(defaults.case_dir.format(case=case_name)).expanduser()
    return path if path.is_absolute() else ROOT / path


def build_system(case_dir: Path, case_name: str, params: Params.Params) -> System_25D:
    options = {
        "filename_blocks": str(case_dir / f"{case_name}.blocks"),
        "filename_nets": str(case_dir / f"{case_name}.nets"),
        "filename_pl": str(case_dir / f"{case_name}.pl"),
    }
    modules, block_headers = parse_blocks(options)
    locations = parse_pls(options)
    nets, net_headers = parse_nets(options)

    num_chiplets = int(block_headers["Headers"]["NumHardRectilinearBlocks"])
    num_terminals = int(block_headers["Headers"]["NumTerminals"])
    system = System_25D(num_chiplets, num_terminals)
    interposer = Passive_Interposer()

    for module_name, module in modules["Modules"].items():
        if "rectangles" in module:
            chiplet = Chiplet(module_name)
            chiplet.set_chiplet_size(*module["rectangles"][0][-2:])
            chiplet.set_chiplet_loc(*module["rectangles"][0][:2])
            system.append_chiplet(module_name, chiplet)
        elif "terminal" in module:
            center = locations["Modules"][module_name]["center"]
            interposer.append_terminal(module_name, center)
            system.append_terminal(module_name, center)

    system.num_nets = int(net_headers["Headers"]["NumNets"])
    system.num_pins = int(net_headers["Headers"]["NumPins"]) - system.num_nodes + num_chiplets

    pin_id = 0
    for net_idx, net in enumerate(nets["Nets"]):
        system.net_id.append(net_idx)
        system.net_weights.append(1.0)
        system.net2pin_map.append([])
        for pin in net:
            node_id = system.node_name2id_map[pin[0]]
            if len(pin) < 3 or pin[2] is None:
                continue
            pin_offset_x = float(pin[1])
            pin_offset_y = float(pin[2])
            existing_pin = None
            for old_pin_id in system.node2pin_map[node_id]:
                if (
                    system.pin_offset_x[old_pin_id] == pin_offset_x
                    and system.pin_offset_y[old_pin_id] == pin_offset_y
                ):
                    existing_pin = old_pin_id
                    break
            if existing_pin is not None:
                system.net2pin_map[net_idx].append(existing_pin)
                system.pin2net_map[existing_pin].append(net_idx)
            else:
                system.net2pin_map[net_idx].append(pin_id)
                system.pin2net_map.append([net_idx])
                system.node2pin_map[node_id].append(pin_id)
                system.pin2node_map.append(node_id)
                system.pin_offset_x.append(pin_offset_x)
                system.pin_offset_y.append(pin_offset_y)
                pin_id += 1

    interposer.set_interposer_size(params.interposer_size)
    fence = [
        params.fence_width,
        interposer.width - params.fence_width,
        params.fence_height,
        interposer.height - params.fence_height,
    ]
    system.set_interposer_size(fence, interposer)
    system.set_bins(params)
    system.num_grid_x = params.num_grid_x
    system.num_grid_y = params.num_grid_y
    system.initialize()
    system.area_cplt = (np.array(system.node_size_x) * np.array(system.node_size_y)).sum()

    system.case_name = case_name
    system.powermap = np.zeros(num_chiplets)
    power_file = case_dir / f"{case_name}.power"
    if power_file.exists():
        with power_file.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) != 2:
                    continue
                name, power = parts
                if name in system.node_name2id_map:
                    system.powermap[system.node_names.index(name)] = float(power)
    return system


def unpack_result(result):
    if isinstance(result, dict):
        hpwl = result["hpwl"]
        best_fp_values = result.get("best_fp_pos", [])
        best_fp = best_fp_values[0] if best_fp_values else None
        return hpwl, best_fp
    hpwl, _best_metric, best_fp = result[:3]
    return hpwl, best_fp


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", required=True, choices=["wl", "thermal"])
    parser.add_argument("--param-file", "--param_file", dest="param_file",
                        help="Case overrides; defaults to WL-driven.json or Thermal-aware.json in opt.case_dir")
    args = parser.parse_args()

    if args.param_file:
        param_file = Path(args.param_file).expanduser().resolve()
    else:
        filename = "WL-driven.json" if args.mode == "wl" else "Thermal-aware.json"
        param_file = default_case_dir(args.case) / filename
    params = load_params(param_file, args.case, mode=args.mode)
    case_dir = Path(params.case_dir)
    out_dir = Path(params.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    system = build_system(case_dir, args.case, params)
    compact_model = build_compact_model(params, system)

    start = time.time()
    result = placeflow_core(params, system, compact_model)
    hpwl, best_fp = unpack_result(result)
    summary = {
        "case": args.case,
        "mode": args.mode,
        "case_dir": str(case_dir),
        "param_file": str(param_file),
        "out_dir": str(out_dir),
        "hpwl": float(hpwl),
        "twl_m": float(hpwl) / 1e6,
        "runtime_s": time.time() - start,
        "has_best_fp": best_fp is not None,
        "thermal_model": params.thermal_model if params.temp_aware_opt else None,
        "thermal_convection_resistance": params.thermal_convection_resistance,
    }
    if best_fp is not None:
        layout = export_layout(best_fp, system, params, args.case, args.mode)
        layout_path = out_dir / "layout.json"
        write_json(layout_path, layout)
        summary["layout_json"] = str(layout_path)
        summary["layout_chiplets"] = len(layout["chiplets"])
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0

