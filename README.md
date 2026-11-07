# ATPlace2.5D

Chiplet placement optimization and thermal-model training. Placement uses editable Python source code. The compact thermal model is fitted to ATSim temperature labels and loaded by the placement optimizer.

## Project structure

```text
run_placement_opt.py        Placement optimization entry point
run_thermal_training.py     Thermal-model training entry point
run_thermal_evaluation.py   Thermal evaluation of an exported layout
params.json                 Global parameter defaults
requirements.txt            Python dependencies
cases/                      Case inputs and parameter overrides
src/
  Params.py                 Configuration loader
  opt/                      Initialization, optimization and legalization
  thermal/                  Thermal model training and thermal evaluation
ATSim/
  ATSim                     Downloaded ATSim executable (not tracked)
  sim.config.template       Simulation settings template
  structure.xml.template    Package structure template
  material.config.template  Material properties template
utils/                      Input parsers and plotting utilities
  visualize_layout.py       Render exported layout.json as an image
  visualize_temp.py         Temperature and placement plotting helpers
results/                    Generated outputs, excluded from version control
```

## Environment setup

The project has been tested with Python 3.9. Dependency versions are pinned in [requirements.txt](requirements.txt). From the project root, install them in your Python environment:

```bash
python -m pip install -r requirements.txt
```

Gurobi is selected by default and requires a valid license. To use CBC, set `opt.ilp_solver` to `"pulp"` in the case configuration. The solver used to sample training layouts can be set separately through `thermal.training_ilp_solver`; its default, `null`, inherits `opt.ilp_solver`.

Download the ATSim Linux executable from [PKU-IDEA/ATSim_pub](https://github.com/PKU-IDEA/ATSim_pub) using the commands in [ATSim/README.md](ATSim/README.md), which also records its version and checksum. The executable is excluded from version control. To use another executable, set `thermal.thermal_solver` to its path and place `sim.config.template`, `structure.xml.template` and `material.config.template` in the same directory. The wrapper generates `sim.config`, `structure.xml`, `material.config` and floorplan/power CSV files in each simulation directory. Layer thicknesses and package structure come from the XML template. Material placeholders are filled by `material_values()` in `src/thermal/atsim.py`, preserving the original material constants and copper-fraction mixture calculations. Case dimensions, paths, grid size, CPU count and thermal boundary settings are filled into template placeholders at runtime; see [ATSim/README.md](ATSim/README.md) for their names and units.

## Configuration

`params.json` has two sections, `opt` and `thermal`, with one parameter level inside each section. Each section has a short `description` string used only as documentation. Each global parameter has a `description` and a `default` value:

- `opt`: placement initialization, continuous optimization and legalization, including case/output directories, solver, thread count and solve time limits.
- `thermal`: thermal-model training, thermal objectives and ATSim evaluation, including simulator/model paths, physical settings, objective parameters (`temp_threshold`, `gamma`, `temp_weight_init`), and `training_*` and `simulation_*` settings.

Section descriptions are metadata: the loader skips them rather than creating runtime parameters. A shortened example of the global defaults format is:

```json
{
  "opt": {
    "description": "Placement initialization, continuous optimization and legalization settings",
    "initialization_time_limit_s": {
      "description": "Time limit in seconds for each initialization ILP attempt",
      "default": 30
    }
  },
  "thermal": {
    "description": "Thermal-model training, thermal objectives and ATSim evaluation settings",
    "thermal_convection_resistance": {
      "description": "Total heatsink-to-ambient convection resistance, K/W",
      "default": 0.01
    }
  }
}
```

Configuration is loaded in this order: **global defaults → case parameter file**. Values in the case file override matching defaults; omitted parameters retain their defaults. Case files contain values directly, without the `description`/`default` wrapper.

Parameter names use lowercase `snake_case`, including `ilp_solver`, `ref_hpwl`, `density_weight_lower_ratio`, `density_weight_upper_ratio`, `density_weight_iterations`, `sub_iterations`, `perturb_flag` and `training_ilp_solver`. Previous spellings are accepted when reading overrides and converted to the current names when saving configuration.

All three entry points accept an optional `--param_file` argument, also spelled `--param-file`. An explicitly supplied file replaces the default case parameter file; the default case file is not loaded in addition to it.

Configured relative paths resolve from the project root. `{case}` expands to the selected case; output directories also support `{timestamp}`, and placement output directories support `{mode}`. Expansion happens at runtime without modifying the configuration file.

## Case format

Each directory under `cases/` describes one chiplet placement problem: chiplet geometry, interposer dimensions, pin connections, fixed terminal locations, power and mode-specific parameters. `Case1`, for example, is selected with `--case Case1` and read from `opt.case_dir` (default `cases/{case}`).

| File | Meaning |
| --- | --- |
| `<Case>.blocks` | Chiplet names and rectangular geometry, terminal declarations, interposer width and height |
| `<Case>.nets` | Net connectivity and each pin's offset from its chiplet center |
| `<Case>.pl` | Placement input coordinates; the current flow uses fixed terminal coordinates and generates chiplet positions through the ILP initializer |
| `<Case>.power` | Chiplet power, one `chiplet_name power_w` pair per line |
| `WL-driven.json` | Default parameter overrides for `--mode wl` |
| `Thermal-aware.json` | Default parameter overrides for `--mode thermal`, thermal-model training and thermal evaluation |

Names must match across the files. Geometry and placement coordinates use micrometers (μm), and power uses watts (W). The `.nets` offsets use percentages rather than μm.

### Geometry and interposer dimensions

Interposer dimensions are read from two additional headers in `<Case>.blocks` for both placement and thermal-model training. For example, `cases/Case1/Case1.blocks` starts with:

```text
NumSoftRectangularBlocks : 0
NumHardRectilinearBlocks : 6
NumTerminals : 0
InterposerWidth : 42000
InterposerHeight : 42000
```

Both dimensions must be finite positive numbers in micrometers. These headers are a project extension to the blocks format, shared by both optimization modes; no case dimensions are hardcoded in the source. Custom cases use the same format. TAP-2.5D stores its dimensions separately in the `[interposer]` section of a `.cfg` file, using `intp_size` in millimeters; this project keeps geometry together in `.blocks`.


A chiplet entry defines its unrotated rectangle through four vertices:

```text
CPU1_0 hardrectilinear 4 (0, 0) (0, 12000) (12000, 12000) (12000, 0)
```

This defines a 12000 × 12000 μm chiplet. `NumHardRectilinearBlocks` counts these chiplets, and `NumTerminals` counts fixed external terminals. The current placement flow uses rectangular hard blocks; soft-block placement is not implemented.

### Connections and pin offsets

A net lists its pin count followed by the participating chiplets and pin offsets:

```text
NetDegree : 2
CPU1_0 B : %49.73333 %-11.29167
HBM_0 B : %47.74375 %-8.66667
```

`B` denotes a bidirectional pin in the input format; the current objective does not distinguish pin directions. For a chiplet of width `W` and height `H`, `%px %py` becomes a local offset `(px * W / 100, py * H / 100)` from the chiplet center. Positive x points right and positive y points up. For the CPU above, the offset is approximately `(5968, -1355)` μm. The optimizer rotates this local offset with the chiplet, then adds the chiplet center to obtain the global pin position. The supplied cases use two-pin nets; the current initialization and legalization formulations assume two-pin connections.

### Adding a case

Create a directory with matching `<Case>.blocks`, `<Case>.nets`, `<Case>.pl` and `<Case>.power` files, then add the two mode-specific JSON files using an existing case as a template. Set the interposer headers in `.blocks` and update chiplet names, geometry, connections and power consistently. Custom cases do not require editing a source-code case list.

## Placement optimization

```bash
python run_placement_opt.py --case Case1 --mode wl
python run_placement_opt.py --case Case1 --mode thermal
```

| Argument | Required | Description |
| --- | --- | --- |
| `--case` | Yes | Case name, such as `Case1` |
| `--mode` | Yes | `wl`: wirelength optimization; `thermal`: thermal-aware optimization |
| `--param_file` | No | Custom parameter override file |

By default, the selected mode loads `WL-driven.json` or `Thermal-aware.json` from `opt.case_dir`, which defaults to `cases/{case}`. See [Case format](#case-format) for input files, units and pin-offset conventions.

Placement outputs are written to `opt.result_dir`:

```text
results/{case}/opt/{mode}_{timestamp}/
  summary.json          Total wirelength, runtime and output paths
  layout.json           Final coordinates, rotated dimensions, angles and power
```

`--mode` determines whether thermal-aware optimization is enabled: `wl` disables it and `thermal` enables it, including when a custom `--param_file` is supplied. `temp_aware_opt` is no longer a configuration option.

Thermal-aware placement requires a trained model and defaults to loading `results/{case}/thermal_model.pt`. Missing checkpoints or incompatible physical settings produce an error rather than an untrained fallback. Model weights are frozen during placement, while gradients with respect to positions and dimensions remain available.

Chiplet reflection is disabled internally. Warpage optimization is not included. Placement uses one stage and preserves the ILP initialization. Position and rotation optimization remain available.

`opt.num_threads` controls CPU threads for continuous optimization and the ILP solvers. Independently, `thermal.num_threads` controls ATSim's `number_of_core` during evaluation and training label generation. Individual ATSim stages can still run serially. Placement and thermal-model training run on the CPU; the CUDA path is incomplete.

Placement prints INFO-level iteration metrics to the console. CBC/Gurobi solver logs are enabled for initialization, legalization and thermal-training layout sampling. ATSim output is saved in each simulation directory’s `stdout.log` and `stderr.log` rather than printed to the console.


Density targets the actual average occupied area, and wirelength uses the implemented orientation-aware formula. Net-degree filtering is not implemented. The former `gpu`, `target_density`, `wirelength`, `ignore_net_degree`, `warp_threshold`, `random_center_init_flag`, `flip_opt`, `warp_aware_opt`, `warp_weight_init`, and `floorplan_stages` options are rejected in configuration files. Unsupported branches are removed or their behavior is fixed in the source.


## Thermal-model training

```bash
python run_thermal_training.py --case Case1
```

Only `--case` is required. Use `--param_file` to select another override file; the default is the case's `Thermal-aware.json`. Training settings come from the `thermal` section:

| Parameter | Default | Description |
| --- | --- | --- |
| `training_samples` | 5 | Number of training layouts |
| `training_test_samples` | 10 | Independent test layouts, excluded from fitting |
| `training_iterations` | 1000 | Number of Adam updates |
| `training_learning_rate` | 0.1 | Initial learning rate |
| `training_seed` | 1234 | Seed for layout sampling and model fitting |
| `training_lr_decay_step` | 100 | Learning-rate decay interval |
| `training_lr_decay_gamma` | 0.95 | Multiplier applied at each decay step |
| `training_ilp_solver` | null | Sampling solver; null inherits the placement solver |
| `training_sampling_time_limit_s` | 30 | Time limit per layout sampling solve, in seconds |
| `training_dataset` | null | Existing dataset path; null generates fresh labels |

The training flow samples legal layouts, generates ATSim labels, fits the analytic thermal model and reports errors on the held-out layouts. When reusing a dataset, its saved training/test split is retained.

Output locations are controlled by `training_result_dir` and `thermal_model`:

```text
results/{case}/
  thermal_model.pt
  training/{timestamp}/
    dataset.pt
    training_report.json
    sample_*/           ATSim inputs, outputs and logs
```

Checkpoint loading verifies the ATSim executable and all three templates, prediction plane, chiplet order, dimensions, power, interposer dimensions, grid and thermal boundary settings. Retrain after changing physical settings, materials, the layer stack or the simulator. Changing placement objective weights alone does not require retraining. Older datasets and checkpoints targeting another solver or the interposer layer are rejected; generate new labels and retrain for the chip-layer midplane.

## Thermal simulation and visualization

Replace `<run>` with an optimization run directory, such as `thermal_20261107_120000`:

```bash
python run_thermal_evaluation.py --layout results/Case1/opt/<run>/layout.json
python utils/visualize_layout.py results/Case1/opt/<run>/layout.json
```

Thermal simulation requires only `--layout`; the case is read from that file. Use `--param_file` to replace the default case `Thermal-aware.json`. Simulation outputs default to `results/{case}/simulation/{timestamp}/` and include the effective configuration, simulator inputs, logs, temperature arrays and metrics.

`utils/visualize_layout.py` renders an exported layout. `utils/visualize_temp.py` provides plotting functions used by the optimization code; it is not a standalone command-line entry point.

Run `run_thermal_evaluation.py` explicitly to evaluate the exported layout. Optimization does not invoke ATSim automatically.

Shared physical settings include `ambient` (45 degrees Celsius by default), `num_grid_x`/`num_grid_y` (64 by 64 by default), and `thermal_convection_resistance`. The resistance is in **K/W**, and ATSim receives `htc_top = 1 / (thermal_convection_resistance * heatsink_top_area)` in W/(m²·K); bottom convection is disabled. The project default is **0.01 K/W**, whereas the paper states 0.1 K/W.

`simulation_timeout_s`, defaulting to 600 seconds, applies to every ATSim call, including training-label generation. Reported maximum temperature is the maximum of the chip-midplane grid. `temperature_c.npy` uses axis order `(x, y)`, with the origin at the lower left of the interposer. `summary.json`, exported layout metadata and saved models identify the physical prediction plane.

### Package model and temperature target

The ATSim route uses one thickness cell per input Bulk/Active layer and a uniform XY grid (`min_depth=max_depth=1`, `tiers_active=tiers_bulk=tiers_chiplet=1`). The heatsink can still be subdivided by `max_thickness`. Chiplet gaps are filled automatically with Underfill; no artificial edge strips or manual whitespace-filling code are used. The former `reso_interposer` option has been removed.

The default bottom-to-top stack in `ATSim/structure.xml.template` is substrate (200 μm), C4 composite (70 μm), silicon/TSV interposer (110 μm), microbump composite (10 μm), silicon chiplets (150 μm), TIM (20 μm), copper spreader (1 mm) and copper heatsink (6.9 mm). The microbump footprints match their chiplets. Spreader side length is the interposer width plus height; heatsink side length is twice that value. Power is distributed uniformly through each chiplet volume.

**Training labels, placement thermal objectives and evaluation all target the chip-layer midplane**, at the height computed from `structure.xml.template`: by default z=465 μm from the package bottom (the chip layer spans z=390–540 μm). ATSim provides the temperatures at these single-tier cell centers directly. The compact model predicts temperature rise above ambient at the same XY centers, and `thermal.temp_threshold` applies to this plane. This is the same layer targeted by the original evaluation flow; numerical values can still differ because the simulator and package discretization have changed.

## Validation

```bash
python -m unittest src.thermal.test_pipeline -v
```

Tests cover configuration loading, checkpoint I/O, input gradients, physical-setting compatibility, ATSim parameters and grid orientation. Evaluate model accuracy against ATSim using independent test layouts; these errors do not measure agreement with physical chip measurements.

`.gitignore` excludes generated results, models, datasets, logs, caches and local environments. No pretrained model is included; train one before running thermal-aware placement.

## Citation

If you use this project in your research, please cite the [ATPlace2.5D paper](https://doi.org/10.1145/3676536.3676648):

```bibtex
@inproceedings{wang2024atplace,
  author    = {Wang, Qipan and Li, Xueqing and Jia, Tianyu and Lin, Yibo and Wang, Runsheng and Huang, Ru},
  title     = {{ATPlace2.5D}: Analytical Thermal-Aware Chiplet Placement Framework for Large-Scale {2.5D-IC}},
  booktitle = {Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design},
  year      = {2024},
  doi       = {10.1145/3676536.3676648},
  url       = {https://doi.org/10.1145/3676536.3676648}
}
```
