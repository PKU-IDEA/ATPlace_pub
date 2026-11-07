# ATSim runtime assets

`ATSim` is the unmodified Linux executable from [PKU-IDEA/ATSim_pub](https://github.com/PKU-IDEA/ATSim_pub), commit `4b4ff818206208c1389151b872977ce0683b8e63`.

The executable is excluded from version control. Download it from the project root:

```bash
curl -fL https://media.githubusercontent.com/media/PKU-IDEA/ATSim_pub/4b4ff818206208c1389151b872977ce0683b8e63/bin/ATSim -o ATSim/ATSim
printf '%s\n' '5350807b3dc082b31293f0f15c5deda390b374e8a0adbfe07567629ed3429eac  ATSim/ATSim' | sha256sum -c -
chmod +x ATSim/ATSim
```

- Upstream file: `bin/ATSim` (Git LFS)
- SHA-256: `5350807b3dc082b31293f0f15c5deda390b374e8a0adbfe07567629ed3429eac`
- `sim.config.template`: simulation defaults adapted from upstream `cases/3DIC/steady.config`.
- `structure.xml.template`: the flat package stack, layer thicknesses and filling materials.
- `material.config.template`: placeholders for material conductivities and volumetric heat capacities, computed by `material_values()` in `src/thermal/atsim.py`.

The wrapper renders these into `sim.config`, `structure.xml` and `material.config` in each run directory, alongside generated floorplan and power CSV files. Runtime geometry and paths use `${interposer_width_m}`, `${interposer_height_m}`, `${spreader_side_m}`, `${sink_side_m}`, `${material_file}`, `${bump_floorplan}`, `${bump_power}`, `${chip_floorplan}` and `${chip_power}` placeholders in the XML template. Lengths are meters. The `Chip` layer name identifies the output plane.

Given interposer width `W` and height `H` from the case `.blocks` file:
- Spreader: side length `W + H`, substituted as `${spreader_side_m}`.
- Heatsink: side length `2 * (W + H)`, substituted as `${sink_side_m}`.

Case dimensions are converted from micrometers to meters before substitution. Both footprints are centered over the interposer by `XYorigin="Center"` in the XML template. For Case1's 42 × 42 mm interposer, the spreader is 84 × 84 mm and the heatsink is 168 × 168 mm. Their thicknesses are independent of interposer dimensions and remain specified in the XML template: 1 mm for the spreader and 6.9 mm for the heatsink.

The simulation template uses `${ambient_k}` for ambient/initial temperature in kelvin, `${num_threads}` for CPU cores, `${num_grid_x}`/`${num_grid_y}` for columns/rows, and `${htc_top}` for top convection. The wrapper computes the latter as `1 / (thermal_convection_resistance * heatsink_top_area)` using the rendered XML geometry.

Material placeholders follow `${<material>_conductivity}` and `${<material>_volumetric_capacity}`, where `<material>` is `substrate`, `underfill`, `silicon`, `tim`, `copper`, `c4`, `tsv` or `ubump`. Conductivity is in W/(m·K), and volumetric heat capacity is in J/(m³·K). C4, TSV and microbump properties retain the original ATPlace mixture rule: copper fraction `f = (diameter / pitch)^2`, with each property calculated as `f * copper + (1 - f) * matrix`. The diameter/pitch pairs are 250/600, 10/50 and 25/45 μm, respectively. C4 and microbumps use Underfill as their matrix; TSV uses Silicon. The original copper heat capacities are preserved: 3,494,400 for the mixtures and 3,550,000 for bulk copper.

The executable is packaged with its Python runtime. See the upstream repository for platform requirements, publications and distribution information.

During evaluation and training label generation, ATSim standard output and errors are saved to `stdout.log` and `stderr.log` in each run directory, rather than printed to the console. `command.json` records the executed command.

The wrapper uses a single thickness tier per input Bulk/Active layer. It extracts the chip-layer midplane directly from cell-center output, deriving its Z coordinate from the XML template (465 μm by default). Empty regions use automatic Underfill filling. Parallel cores follow `thermal.num_threads` independently of optimization, and convection uses the rendered heatsink top area. All three templates are included in dataset/checkpoint compatibility checks.
