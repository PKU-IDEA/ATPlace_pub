# ATPlace2.5D Public Package

This repository contains a compact public package for ATPlace2.5D (ICCAD 2024).

## Contents

- `cases/`: ten clean case directories. Each case contains Bookshelf-style input files, `WL-driven.json`, and `Thermal-aware.json`.
- `src/`: encrypted ATPlace layout kernel and its runtime.
- `thermal/`: HotSpot files used by thermal evaluation.
- `utils/`: parsers for case input files.
- `Thermal.py`: thermal helper entry file.
- `reproduce.sh`: shell entry for selecting a case and placement mode and starting the encrypted layout kernel.

## Usage

Create a Python environment with the package dependencies used by the encrypted kernel. `gurobipy` is the recommended solver interface, and the parameter files set `ILPsolver` to `grb`.

Each case directory contains Bookshelf input files and two layout parameter files.

- `WL-driven.json` is for wirelength-driven placement.
- `Thermal-aware.json` is for thermal-aware placement.

`pulp` may work in some environments, but it is not guaranteed to produce strictly identical legal placement or numerical values. Use `gurobipy` when reproducing reported layouts.

On a Linux server, activate the reproduction environment first:

```bash
conda activate ATPlan
```

Run one case directly through the shell entry:

```bash
bash reproduce.sh Case1 wl
bash reproduce.sh Case1 thermal
```

The first argument is one of `Case1` through `Case10`. The second argument is `wl` or `thermal`.

The command selects:

- input files from `cases/<CaseX>/`;
- `WL-driven.json` for `wl`;
- `Thermal-aware.json` for `thermal`;
- output directory `results/<CaseX>_<mode>_<timestamp>/` unless `ATPLACE_OUT_DIR` is set.

Then it calls `reproduce.py`, which loads the selected Bookshelf files and parameter file and invokes the encrypted `ATPLACE.PlaceFlow.placeflow_core` kernel. A successful execution writes:

- `summary.json`: selected case, mode, parameter file, output directory, HPWL/TWL, runtime, and whether a final layout was returned;
- `layout.json`: final chiplet coordinates and chiplet sizes when the encrypted kernel returns a final layout.

Optional environment variables:

```bash
PYTHON=/path/to/python bash reproduce.sh Case1 wl
ATPLACE_OUT_DIR=/tmp/case1_wl bash reproduce.sh Case1 wl
ATPLACE_THERMAL_DIR=/path/to/thermal bash reproduce.sh Case1 thermal
ATPLACE_DRY_RUN=1 bash reproduce.sh Case1 wl
```

The case parameter files are self-contained layout settings. Do not change interposer geometry unless the target design itself changes.

## Included Parameter Files

The public package includes one `WL-driven.json` and one `Thermal-aware.json` for each case. These files are ordinary reproduction settings for the encrypted kernel; they are not Optuna/HPO logs and do not include experiment tables.

The current public parameter files include explicit local representative settings for these successfully reproduced modes:

- thermal-aware: `Case1`, `Case2`, `Case5`, `Case7`, `Case9`, `Case10`;
- wirelength-driven: `Case3`, `Case4`, `Case8`, `Case9`, `Case10`.

Some modes in the package use preserved baseline settings because the corresponding accepted representative came from a saved layout rather than a standalone parameter record. The remaining thermal-aware search targets at the time of this package update are `Case3`, `Case6`, and `Case8`.

## Thermal-Aware Mode

Thermal-aware placement uses two thermal components.

1. `reproduce.py` creates a compact analytic thermal model object for the selected case and passes it into the encrypted placement kernel. The public wrapper does not run a separate visible thermal-model training program and does not write a pretrained thermal-model checkpoint before placement starts.
2. `thermal/` contains the HotSpot executable and the default HotSpot configuration used by the public run.
3. `Thermal.py` is a readable reference/helper file for HotSpot-style floorplan, layer, power-trace, and configuration generation. The default public wrapper does not import this file directly, so adding print statements to `Thermal.py` is not expected to change the default `bash reproduce.sh Case1 thermal` console output.

The main thermal-aware layout controls are in `cases/<CaseX>/Thermal-aware.json`:

- `temp_aware_opt`: enables the thermal-aware objective;
- `temp_weight_init`, `temp_threshold`, and `gamma`: control the thermal penalty;
- `wl_weight`: controls the wirelength term in the thermal-aware objective;
- `target_density`, `density_weight_init`, `overflow_init`, and `eta`: control placement density and optimization behavior;
- `floorplan_stages[0].learning_rate`: position and angle learning rates.

Use these JSON files for ordinary parameter changes. Keep each case's `interposer_size` unchanged unless the physical case definition changes.

## Changing Thermal Configuration

For a new HotSpot configuration, copy the thermal directory and point the run to it:

```bash
cp -R thermal thermal_custom
# edit thermal_custom/hotspot.config
ATPLACE_THERMAL_DIR="$PWD/thermal_custom" bash reproduce.sh Case1 thermal
```

Changing `thermal_custom/hotspot.config` is enough for ordinary HotSpot parameter tests. If the stack definition or generated HotSpot files must change, use `Thermal.py` as the readable starting point. Examples include changing layer material constants, layer thicknesses, bump parameters, or how the HotSpot `.flp`, `.lcf`, and `.ptrace` files are generated. When doing this, keep the encrypted placement kernel unchanged and document which generated thermal files changed.

For a new case, create `cases/<NewCase>/` with matching Bookshelf-style files:

```text
<NewCase>.blocks
<NewCase>.nets
<NewCase>.pl
<NewCase>.power
WL-driven.json
Thermal-aware.json
```

Then either add the case interposer size to `CASE_INTERPOSER_SIZE` in `reproduce.py`, or include `interposer_size` in the case parameter JSON.

## Layout Visualization

After a successful run, generate a PNG from the exported layout:

```bash
python visualize_layout.py results/Case1_wl_YYYYMMDD_HHMMSS/layout.json
python visualize_layout.py results/Case1_thermal_YYYYMMDD_HHMMSS/layout.json --out case1_thermal.png
```

The visualizer reads only `layout.json`. It does not call the encrypted placement kernel.

## Citation

```bibtex
@inproceedings{wang2024atplace,
  title={ATPlace2.5D: Analytical Thermal-Aware Chiplet Placement Framework for Large-Scale 2.5D-IC},
  author={Wang, Qipan and Li, Xueqing and Jia, Tianyu and Lin, Yibo and Wang, Runsheng and Huang, Ru},
  booktitle={Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design},
  pages={1--9},
  year={2024}
}
```
