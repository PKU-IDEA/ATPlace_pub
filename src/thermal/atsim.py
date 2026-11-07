"""ATSim steady-state package generation and chip-midplane temperature extraction."""
import configparser
import csv
import json
import os
from pathlib import Path
import signal
from string import Template
import subprocess
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

DEFAULT_ASSETS = Path(__file__).resolve().parents[2] / 'ATSim'


def chip_layer_bounds(structure_template=None):
    """Read the chip layer's physical Z bounds from the flat package template."""
    path = Path(structure_template) if structure_template is not None else DEFAULT_ASSETS / 'structure.xml.template'
    package = ET.parse(path).getroot()
    if package.tag != 'Package' or package.get('lengthunit') != 'm' or package.findall('.//Chiplet'):
        raise ValueError(f'{path}: expected a flat package stack in meters')
    bottom = 0.
    bounds = []
    for layer in package.findall('./Component/Layer'):
        geometry = layer.find('Geometry')
        thickness = float(geometry.get('Thickness')) if geometry is not None else float('nan')
        if not np.isfinite(thickness) or thickness <= 0:
            raise ValueError(f'{path}: layer thicknesses must be finite positive numbers in meters')
        if layer.get('name') == 'Chip':
            bounds.append((bottom, bottom + thickness))
        bottom += thickness
    if len(bounds) != 1:
        raise ValueError(f'{path}: expected exactly one layer named Chip')
    return bounds[0]


def temperature_target(structure_template=None):
    """Physical prediction plane shared by training, evaluation and checkpoints."""
    bottom, top = chip_layer_bounds(structure_template)
    return {'name': 'chip_midplane', 'z_m': (bottom + top) / 2,
            'extraction': 'single_tier_cell_center', 'axis_order': 'xy',
            'origin': 'interposer_lower_left', 'unit': 'C'}


def material_values():
    """Retain the original material constants and square-area copper homogenization."""
    properties = {
        'substrate': (1 / 3.33, 1.06e6), 'underfill': (1 / .625, 2.32e6),
        'silicon': (1 / .01, 1.75e6), 'tim': (1 / .25, 4e6),
        'copper': (400., 3.55e6),
    }
    # The original composite calculations use this copper capacity, distinct from bulk copper above.
    composite_copper_capacity = 3494400.
    for name, diameter_um, pitch_um, matrix in (
        ('c4', 250., 600., 'underfill'), ('tsv', 10., 50., 'silicon'),
        ('ubump', 25., 45., 'underfill'),
    ):
        fraction = (diameter_um / pitch_um) ** 2
        conductivity, capacity = properties[matrix]
        properties[name] = (fraction * 400. + (1 - fraction) * conductivity,
                            fraction * composite_copper_capacity + (1 - fraction) * capacity)
    return {f'{name}_{key}': value for name, values in properties.items()
            for key, value in zip(('conductivity', 'volumetric_capacity'), values)}


def render_template(path, values):
    try:
        return Template(Path(path).read_text()).substitute(values)
    except KeyError as exc:
        raise ValueError(f'Unknown placeholder in {path}: {exc.args[0]}') from exc


class ATSimSolver:
    def __init__(self, work_dir, params):
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.params = params
        self.binary = Path(params.thermal_solver).resolve()
        self.template = self.binary.parent / 'sim.config.template'
        self.structure_template = self.binary.parent / 'structure.xml.template'
        self.material_template = self.binary.parent / 'material.config.template'
        for path in (self.binary, self.template, self.structure_template, self.material_template):
            if not path.is_file():
                raise FileNotFoundError(f'ATSim executable/template missing: {path}')
        self.chip_bounds = chip_layer_bounds(self.structure_template)
        self.temperature_target = temperature_target(self.structure_template)
        for name in ('thermal_convection_resistance', 'simulation_timeout_s'):
            value = getattr(params, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('num_grid_x', 'num_grid_y', 'thermal_num_threads'):
            value = getattr(params, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not np.isfinite(params.ambient):
            raise ValueError('ambient must be finite, in degrees Celsius')

    def set_params(self, system):
        self.num_chiplets = system.num_chiplets
        self.width = float(system.intp_width) * 1e-6
        self.height = float(system.intp_height) * 1e-6
        if not np.isfinite([self.width, self.height]).all() or min(self.width, self.height) <= 0:
            raise ValueError('Interposer dimensions must be finite and positive')

    def set_pos(self, power, pos, width, height):
        # Inputs are chiplet-center coordinates in micrometers; ATSim uses meters.
        self.power = np.asarray(power, dtype=float)
        self.x, self.y, self.chip_width, self.chip_height = (
            np.asarray(a, dtype=float) * 1e-6 for a in (pos[0], pos[1], width, height))
        arrays = (self.power, self.x, self.y, self.chip_width, self.chip_height)
        if any(a.shape != (self.num_chiplets,) or not np.isfinite(a).all() for a in arrays):
            raise ValueError('Invalid chiplet geometry or power arrays')
        if (self.power < 0).any() or (self.chip_width <= 0).any() or (self.chip_height <= 0).any():
            raise ValueError('Power must be nonnegative and chiplet dimensions positive')
        tolerance = 2e-9  # Accommodate float32 placement export, without shifting chiplets.
        for center, size, limit in ((self.x, self.chip_width, self.width),
                                    (self.y, self.chip_height, self.height)):
            if (center - size / 2 < -tolerance).any() or (center + size / 2 > limit + tolerance).any():
                raise ValueError('Chiplet extends outside the interposer')

    def generate_inputs(self):
        w, h = self.width, self.height
        for prefix, material in (('Bump', 'Ubump'), ('Chip', 'Silicon')):
            floorplan = self.work_dir / f'{prefix}.csv'
            power = self.work_dir / f'{prefix}_power.csv'
            names = [f'{prefix}_{i}' for i in range(self.num_chiplets)]
            with floorplan.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['UnitName', 'X', 'Y', 'Z', 'Length', 'Width', 'Thickness', 'Label'])
                for i, name in enumerate(names):
                    writer.writerow([name, self.x[i] - self.chip_width[i] / 2,
                                     self.y[i] - self.chip_height[i] / 2, 0,
                                     self.chip_width[i], self.chip_height[i], '', material])
            with power.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(names)
                writer.writerow(self.power if prefix == 'Chip' else np.zeros(self.num_chiplets))
        values = {
            'interposer_width_m': w, 'interposer_height_m': h,
            'spreader_side_m': w + h, 'sink_side_m': 2 * (w + h),
            'material_file': self.work_dir / 'material.config',
            'bump_floorplan': self.work_dir / 'Bump.csv', 'bump_power': self.work_dir / 'Bump_power.csv',
            'chip_floorplan': self.work_dir / 'Chip.csv', 'chip_power': self.work_dir / 'Chip_power.csv',
        }
        # Substitute attribute values through ElementTree so paths containing XML characters remain valid.
        package = ET.parse(self.structure_template).getroot()
        for node in package.iter():
            for key, value in list(node.attrib.items()):
                try:
                    node.set(key, Template(value).substitute(values))
                except KeyError as exc:
                    raise ValueError(f'Unknown structure template placeholder: {exc.args[0]}') from exc
        sinks = package.findall("./Component[@type='Sink'][@sinktype='nopackage']/Geometry")
        if len(sinks) != 1:
            raise ValueError('Structure template must define one nopackage heatsink')
        sink_width, sink_height = float(sinks[0].get('Length')), float(sinks[0].get('Width'))
        if not np.isfinite([sink_width, sink_height]).all() or min(sink_width, sink_height) <= 0:
            raise ValueError('Heatsink dimensions must be finite and positive')
        ET.indent(package)
        ET.ElementTree(package).write(self.work_dir / 'structure.xml', encoding='utf-8', xml_declaration=True)
        (self.work_dir / 'material.config').write_text(render_template(self.material_template, material_values()))
        simulation_values = {
            'htc_top': 1 / (self.params.thermal_convection_resistance * sink_width * sink_height),
            'ambient_k': self.params.ambient + 273.15,
            'num_threads': self.params.thermal_num_threads,
            'num_grid_x': self.params.num_grid_x, 'num_grid_y': self.params.num_grid_y,
        }
        config = configparser.ConfigParser()
        config.read_string(render_template(self.template, simulation_values))
        settings = {
            'NoPackage': {'htc_bot': 0},
            'Simulation': {'sim_mode': 0,
                           'calc_leakage': 0, 'cycles': 1, 'power_type': 'value'},
            # A single uniform tier gives labels at exactly the analytic model's XY cell centers.
            'MeshConfig': {'min_depth': 1, 'max_depth': 1, 'tiers_active': 1,
                           'tiers_bulk': 1, 'tiers_chiplet': 1},
        }
        for section, values in settings.items():
            if not config.has_section(section):
                config.add_section(section)
            for key, value in values.items():
                config.set(section, key, str(value))
        with (self.work_dir / 'sim.config').open('w') as handle:
            config.write(handle)
        return self.work_dir / 'structure.xml', self.work_dir / 'sim.config'

    def read_chip_temperature(self, path):
        raw = np.load(path, allow_pickle=False)
        if raw.ndim != 2 or raw.shape[1] != 7 or not np.isfinite(raw).all():
            raise ValueError('Invalid ATSim temperature field; expected finite Nx7 cell data')
        layer_bottom, layer_top = self.chip_bounds
        middle = self.temperature_target['z_m']
        cells = raw[(raw[:, 4] < middle) & (raw[:, 5] > middle)]
        nx, ny = self.params.num_grid_x, self.params.num_grid_y
        if len(cells) != nx * ny:
            raise ValueError(f'ATSim chip layer has {len(cells)} cells; expected {nx * ny}')
        # Use geometry, never file order, to establish the lower-left XY origin.
        x = (cells[:, 0] + cells[:, 1]) / 2
        y = (cells[:, 2] + cells[:, 3]) / 2
        ix = np.floor(x / self.width * nx).astype(int)
        iy = np.floor(y / self.height * ny).astype(int)
        if ((ix < 0) | (ix >= nx) | (iy < 0) | (iy >= ny)).any() or len(np.unique(ix * ny + iy)) != nx * ny:
            raise ValueError('ATSim chip cells do not cover the expected XY grid')
        if not (np.allclose(x, (ix + .5) * self.width / nx, rtol=1e-6, atol=1e-10)
                and np.allclose(y, (iy + .5) * self.height / ny, rtol=1e-6, atol=1e-10)
                and np.allclose(cells[:, 4], layer_bottom, atol=1e-12, rtol=1e-6)
                and np.allclose(cells[:, 5], layer_top, atol=1e-12, rtol=1e-6)):
            raise ValueError('ATSim chip mesh does not match the single-tier uniform label grid')
        result = np.empty((nx, ny), dtype=float)
        result[ix, iy] = cells[:, 6] - 273.15
        return result

    def run(self):
        xml_path, config_path = self.generate_inputs()
        output = self.work_dir / 'outputs'
        raw_path = output / 'raw' / 'steadytemp.npy'
        # Do not accidentally accept output from a prior run in a reused directory.
        raw_path.unlink(missing_ok=True)
        cmd = [str(self.binary), '-xml', str(xml_path), '-config', str(config_path),
               '--output_path', str(output), '--plot_flag', '0']
        (self.work_dir / 'command.json').write_text(json.dumps(cmd, indent=2) + '\n')
        env = dict(os.environ)
        # ATSim controls its workers; pass the thermal CPU setting to numerical libraries too.
        env.update(OMP_NUM_THREADS=str(self.params.thermal_num_threads), MKL_NUM_THREADS=str(self.params.thermal_num_threads),
                   OPENBLAS_NUM_THREADS=str(self.params.thermal_num_threads))
        start = time.time()
        with (self.work_dir / 'stdout.log').open('w') as stdout, (self.work_dir / 'stderr.log').open('w') as stderr:
            try:
                process = subprocess.Popen(cmd, cwd=self.work_dir, env=env, stdout=stdout, stderr=stderr,
                                           start_new_session=True)
                try:
                    process.wait(timeout=self.params.simulation_timeout_s)
                except subprocess.TimeoutExpired:
                    # ATSim may have child workers; terminate the complete simulation group.
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, cmd)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(f'ATSim failed; see {self.work_dir / "stdout.log"} and {self.work_dir / "stderr.log"}') from exc
        self.runtime_s = time.time() - start
        return self.read_chip_temperature(raw_path)


def evaluate_layout(layout, params, out_dir):
    solver = ATSimSolver(out_dir, params)
    chips = layout['chiplets']
    solver.set_params(SimpleNamespace(num_chiplets=len(chips), intp_width=layout['interposer']['width'],
                                      intp_height=layout['interposer']['height']))
    solver.set_pos([c['power_w'] for c in chips],
                   [[c['x'] for c in chips], [c['y'] for c in chips]],
                   [c['width'] for c in chips], [c['height'] for c in chips])
    temperature = solver.run()
    np.save(solver.work_dir / 'temperature_c.npy', temperature)
    result = {'solver': 'ATSim', 'tmax_c': float(temperature.max()), 'tmin_c': float(temperature.min()),
              'runtime_s': solver.runtime_s, 'ambient_c': params.ambient,
              'thermal_convection_resistance': params.thermal_convection_resistance,
              'temperature_target': solver.temperature_target,
              'num_threads': params.thermal_num_threads, 'grid': [params.num_grid_x, params.num_grid_y]}
    (solver.work_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
