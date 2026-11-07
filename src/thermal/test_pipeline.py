"""Contract checks for saved models, physical settings, and grid orientation."""
import copy
import configparser
import csv
import xml.etree.ElementTree as ET
import subprocess
import json
import os
from pathlib import Path
import tempfile
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import numpy as np
import torch

from .model import AnalyticThermalModel
from .checkpoint import metadata, load_model, new_model
from .atsim import ATSimSolver, chip_layer_bounds, temperature_target
from .training import validate_dataset, fit_model

ROOT = Path(__file__).resolve().parents[2]


class ThermalPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.params = SimpleNamespace(thermal_solver=str(ROOT/'ATSim/ATSim'),
            thermal_convection_resistance=.01, ambient=45., num_grid_x=8,
            num_grid_y=4, num_threads=1, thermal_num_threads=1,
            training_samples=1, training_test_samples=1, training_iterations=20,
            training_learning_rate=.1, training_seed=1234, training_lr_decay_step=100,
            training_lr_decay_gamma=.95, training_sampling_time_limit_s=30,
            training_ilp_solver=None, training_dataset=None, training_result_dir=str(self.out),
            simulation_timeout_s=600, simulation_result_dir=str(self.out))
        self.system = SimpleNamespace(case_name='unit', intp_width=20000., intp_height=16000.,
            num_chiplets=2, num_nodes=2, node_names=['A','B'],
            node_size_x=np.array([4000.,5000.]), node_size_y=np.array([3000.,4000.]),
            powermap=np.array([10.,20.]))

    def save(self):
        model = new_model(self.params,self.system)
        path=self.out/'model.pt'
        torch.save({'metadata':metadata(self.params,self.system),'state_dict':model.state_dict()},path)
        return path,model

    def test_runtime_imports_opt_package(self):
        import sys
        sys.path.insert(0,str(ROOT))
        from src.opt import cli as placement
        module=sys.modules[placement.placeflow_core.__module__]
        self.assertEqual(Path(module.__file__).resolve().parent,ROOT/'src/opt')

    def test_root_defaults_and_removed_unused_parameters(self):
        import sys
        sys.path.insert(0,str(ROOT))
        from src.Params import Params
        params=Params()
        self.assertEqual(params.thermal_convection_resistance,.01)
        for key in ['aux_input','lef_input','def_input','density_weight','gp_noise_ratio','dtype']:
            self.assertFalse(hasattr(params,key))
        self.assertFalse((ROOT/'src/params.json').exists())

    def test_independent_optimization_and_thermal_threads(self):
        from src.Params import Params
        params = Params()
        params.fromJson({'opt': {'num_threads': 2}, 'thermal': {'num_threads': 8}})
        self.assertEqual(params.num_threads, 2)
        self.assertEqual(params.thermal_num_threads, 8)
        params.dump(self.out/'threads.json')
        restored = Params()
        restored.load(self.out/'threads.json')
        self.assertEqual(restored.num_threads, 2)
        self.assertEqual(restored.thermal_num_threads, 8)
        self.assertEqual(restored.toJson()['thermal']['num_threads'], 8)
        self.assertNotIn('thermal_num_threads', restored.toJson()['thermal'])

    def test_parameter_names_and_legacy_overrides(self):
        from src.Params import Params
        params = Params()
        for section, fields in params.toJson().items():
            for key in fields:
                self.assertRegex(key, r'^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$')
        for old, new in Params._RENAMED_OPTIONS.items():
            section = params._sections[new]
            value = 'pulp' if new.endswith('ilp_solver') else 7
            params.fromJson({section: {old: value}})
            self.assertEqual(getattr(params, new), value)
            self.assertNotIn(old, params.toJson()[section])
            self.assertFalse(hasattr(params, old))
            with self.assertRaisesRegex(ValueError, 'Duplicate parameter'):
                Params().fromJson({section: {old: value, new: value}})

    def test_flat_sections_merge_and_roundtrip(self):
        import sys
        sys.path.insert(0,str(ROOT))
        from src.Params import Params
        params=Params()
        self.assertEqual(set(params.toJson()),{'opt','thermal'})
        params.fromJson({'thermal':{'training_iterations':7,'simulation_timeout_s':23},
                         'opt':{'learning_rate':[3,.01]}})
        self.assertEqual(params.training_iterations,7)
        self.assertEqual(params.training_samples,5)
        self.assertEqual(params.simulation_timeout_s,23)
        params.dump(self.out/'params.json')
        restored=Params();restored.load(self.out/'params.json')
        self.assertEqual(restored.toJson(),params.toJson())
        with self.assertRaisesRegex(ValueError,'belongs'):
            params.fromJson({'opt':{'training_samples':2}})
        root=json.loads((ROOT/'params.json').read_text())
        self.assertFalse(hasattr(params, 'description'))
        params.fromJson({'opt': {'description': 'Placement settings'},
                         'thermal': {'description': 'Thermal settings'}})
        self.assertFalse(hasattr(params, 'description'))
        for fields in root.values():
            self.assertIsInstance(fields['description'], str)
            entries = [entry for key, entry in fields.items() if key != 'description']
            self.assertTrue(all('default' in entry for entry in entries))
            self.assertTrue(all(not isinstance(entry['default'],dict) for entry in entries))
        self.assertNotIn('floorplan_stages',root['opt'])

    def test_all_cases_flat_opt_rebuilds_kernel_stage(self):
        import sys
        sys.path.insert(0,str(ROOT))
        from src.opt import cli as placement
        for case_dir in sorted((ROOT/'cases').iterdir()):
            if not (case_dir/(case_dir.name+'.blocks')).is_file():
                continue
            case = case_dir.name
            for mode in ['WL-driven','Thermal-aware']:
                path=ROOT/'cases'/case/(mode+'.json')
                data=json.loads(path.read_text())
                self.assertEqual(set(data),{'opt','thermal'})
                self.assertNotIn('floorplan_stages',data['opt'])
                params=placement.load_params(path,case,self.out, mode='wl' if mode == 'WL-driven' else 'thermal')
                self.assertEqual(params.temp_aware_opt, mode == 'Thermal-aware')
                from utils.blocks_parser import parse_blocks
                _, headers = parse_blocks({'filename_blocks': str(case_dir/(case+'.blocks'))})
                self.assertEqual(params.interposer_size, [headers['Headers']['InterposerWidth'], headers['Headers']['InterposerHeight']])
                self.assertEqual(params.floorplan_stages[0]['learning_rate'],data['opt']['learning_rate'])
                self.assertEqual(params.floorplan_stages[0]['optimizer'],data['opt']['optimizer'])

    def test_fixed_options_are_rejected_and_case_configs_are_clean(self):
        from src.Params import Params
        for key in Params._REMOVED_OPTIONS:
            with self.subTest(key=key):
                self.assertFalse(hasattr(Params(), key))
                with self.assertRaisesRegex(ValueError, 'no longer configurable'):
                    Params().fromJson({'opt': {key: 1}})
                with self.assertRaisesRegex(ValueError, 'no longer configurable'):
                    Params().fromJson({key: 1})
        for path in (ROOT/'cases').glob('*/*.json'):
            data = json.loads(path.read_text())
            for fields in data.values():
                self.assertFalse(set(fields) & Params._REMOVED_OPTIONS, str(path))

    def test_placement_uses_cpu_ilp_positions_and_actual_density(self):
        from src.opt import cli as placement
        from src.opt.nonlinear_place import NonlinearPlace
        params = placement.load_params(ROOT/'cases/Case1/WL-driven.json', 'Case1', self.out)
        params.num_threads = 1
        system = placement.build_system(ROOT/'cases/Case1', 'Case1', params)
        # Directly supplied legacy attributes cannot change the fixed behavior either.
        params.gpu = 1
        params.random_center_init_flag = 1
        params.target_density = .001
        placer = NonlinearPlace(params, system)
        self.assertEqual(placer.device.type, 'cpu')
        np.testing.assert_array_equal(placer.init_pos[:system.num_nodes], system.node_x)
        np.testing.assert_array_equal(placer.init_pos[system.num_nodes:], system.node_y)
        data = placer.datacollection
        expected = data.tot_mov_areas/(data.xhigh-data.xlow)/(data.yhigh-data.ylow)
        self.assertAlmostEqual(data.target_dens, expected.item())
        self.assertEqual(len(params.floorplan_stages), 1)
        self.assertNotIn('wirelength', params.floorplan_stages[0])

    def test_optimization_mode_controls_thermal_flag_and_model_loading(self):
        from src.opt import cli as placement
        sentinel = object()
        for filename in ['WL-driven.json', 'Thermal-aware.json']:
            for mode in ['wl', 'thermal']:
                with self.subTest(filename=filename, mode=mode):
                    params = placement.load_params(ROOT/'cases/Case1'/filename, 'Case1', mode=mode)
                    self.assertEqual(params.temp_aware_opt, mode == 'thermal')
                    self.assertNotIn('temp_aware_opt', params.toJson()['opt'])
                    self.assertIn('/opt/' + mode + '_', params.result_dir)
                    with patch('src.thermal.checkpoint.load_model', return_value=sentinel) as loader:
                        result = placement.build_compact_model(params, self.system)
                        if mode == 'thermal':
                            self.assertIs(result['Thermal'], sentinel)
                            loader.assert_called_once_with(params.thermal_model, params, self.system)
                        else:
                            loader.assert_not_called()
                            self.assertFalse(result)
        with self.assertRaisesRegex(ValueError, 'Unsupported optimization mode'):
            placement.load_params(ROOT/'cases/Case1/WL-driven.json', 'Case1', mode='invalid')
        # Training and simulation do not need an optimization mode.
        params = placement.load_params(ROOT/'cases/Case1/Thermal-aware.json', 'Case1', self.out)
        self.assertFalse(params.temp_aware_opt)

    def test_interposer_size_comes_from_configured_case_directory(self):
        from src.opt import cli as placement
        case_dir = self.out/'CustomCase'
        case_dir.mkdir()
        blocks_path = case_dir/'CustomCase.blocks'
        blocks_path.write_text('InterposerWidth : 21000\nInterposerHeight : 17000\n')
        config = self.out/'override.json'
        config.write_text(json.dumps({'opt': {'case_dir': str(case_dir)}}))
        params = placement.load_params(config, 'CustomCase', self.out, mode='wl')
        self.assertEqual(params.interposer_size, [21000, 17000])
        for header in ['', 'InterposerWidth : 1\n',
                       'InterposerWidth : -1\nInterposerHeight : 2\n',
                       'InterposerWidth : nan\nInterposerHeight : 2\n',
                       'InterposerWidth : 1\nInterposerHeight : inf\n']:
            blocks_path.write_text(header)
            with self.assertRaisesRegex(ValueError, 'InterposerWidth and InterposerHeight'):
                placement.load_params(config, 'CustomCase', self.out, mode='wl')
        blocks_path.unlink()
        with self.assertRaises(FileNotFoundError):
            placement.load_params(config, 'CustomCase', self.out, mode='wl')


    def test_model_formula_and_input_gradient_regression(self):
        # Captured model values verify that reorganizing modules preserves the formula.
        model=new_model(self.params,self.system)
        x=torch.tensor([[5000.,14000.]],requires_grad=True)
        args=(x,torch.tensor([[4000.,11000.]]),torch.tensor([[4000.,5000.]]),torch.tensor([[3000.,4000.]]),torch.tensor([[10.,20.]]))
        field=model(args)
        expected=torch.tensor([[3.7549099922,3.0006227493],[4.3980274200,5.3527803421],[3.0460910797,12.4025192261]])
        torch.testing.assert_close(field[0,0,::3,::2],expected)
        expected_grad=torch.tensor([[.0013819433516,-.0021871591453]])
        torch.testing.assert_close(torch.autograd.grad(field.sum(),x)[0],expected_grad)

    def test_project_paths_are_independent_of_working_directory(self):
        import sys
        sys.path.insert(0,str(ROOT))
        from src.opt import cli as placement
        previous=Path.cwd()
        try:
            os.chdir(self.out)
            params=placement.load_params(ROOT/'cases/Case1/Thermal-aware.json','Case1',self.out)
        finally:
            os.chdir(previous)
        self.assertEqual(Path(params.case_dir),ROOT/'cases/Case1')
        self.assertEqual(Path(params.thermal_solver),ROOT/'ATSim/ATSim')
        self.assertFalse(hasattr(params,'thermal_dir'))
        self.assertEqual(Path(params.thermal_model),ROOT/'results/Case1/thermal_model.pt')

    def test_explicit_solver_path_uses_adjacent_template_and_exact_binary(self):
        assets=self.out/'custom solver'
        assets.mkdir()
        binary=assets/'renamed-atsim'
        shutil.copy2(ROOT/'ATSim/ATSim',binary)
        for name in ['sim.config.template','structure.xml.template','material.config.template']:
            shutil.copy2(ROOT/'ATSim'/name,assets/name)
        self.params.thermal_solver=str(binary)
        solver=ATSimSolver(self.out/'work',self.params)
        self.assertEqual(solver.binary,binary)
        self.assertEqual(solver.template,assets/'sim.config.template')
        signature=metadata(self.params,self.system)
        self.assertEqual(signature['atsim_binary_sha256'],metadata(SimpleNamespace(**{**vars(self.params),'thermal_solver':str(ROOT/'ATSim/ATSim')}),self.system)['atsim_binary_sha256'])

    def custom_assets(self):
        assets = self.out/'solver assets'
        assets.mkdir()
        binary = assets/'test-solver'
        binary.write_bytes(b'ATSim template unit-test fixture')
        for name in ['sim.config.template','structure.xml.template','material.config.template']:
            shutil.copy2(ROOT/'ATSim'/name, assets/name)
        self.params.thermal_solver = str(binary)
        return assets

    def test_structure_and_material_templates_drive_generated_model(self):
        assets = self.custom_assets()
        old_checkpoint, _ = self.save()
        tree = ET.parse(assets/'structure.xml.template')
        chip = tree.find("./Component/Layer[@name='Chip']/Geometry")
        chip.set('Thickness','160e-6')
        sink = tree.find("./Component[@sinktype='nopackage']/Geometry")
        sink.set('Length','.1');sink.set('Width','.12')
        tree.write(assets/'structure.xml.template')
        material = configparser.ConfigParser();material.read(assets/'material.config.template')
        material['Silicon']['conductivity (w/(m-k))'] = '140'
        with (assets/'material.config.template').open('w') as handle:material.write(handle)
        solver = ATSimSolver(self.out/'run & data',self.params)
        solver.set_params(self.system)
        solver.set_pos(self.system.powermap,[np.array([4000.,13000.]),np.array([4000.,10000.])],
                       self.system.node_size_x,self.system.node_size_y)
        xml, config = solver.generate_inputs()
        self.assertAlmostEqual(solver.temperature_target['z_m'],470e-6)
        generated = ET.parse(xml)
        self.assertEqual(generated.find('MaterialLib').get('File'),str(solver.work_dir/'material.config'))
        self.assertEqual(generated.find("./Component/Layer[@name='Chip']/Geometry").get('Thickness'),'160e-6')
        settings = configparser.ConfigParser();settings.read(config)
        self.assertAlmostEqual(settings.getfloat('NoPackage','htc_top'),1/(.01*.1*.12))
        generated_material = configparser.ConfigParser();generated_material.read(solver.work_dir/'material.config')
        self.assertEqual(generated_material.getfloat('Silicon','conductivity (w/(m-k))'),140.)
        with self.assertRaisesRegex(ValueError,'mismatch'):
            load_model(old_checkpoint,self.params,self.system)

    def test_rendered_materials_preserve_original_properties(self):
        solver = self.solver()
        solver.generate_inputs()
        material = configparser.ConfigParser();material.read(self.out/'material.config')
        expected = {'Substrate':(.3003003003003003,1060000.), 'Underfill':(1.6,2320000.),
                    'Silicon':(100.,1750000.), 'TIM':(4.,4000000.), 'Copper':(400.,3550000.),
                    'C4':(70.76666666666668,2523888.888888889), 'TSV':(112.,1819776.),
                    'Ubump':(124.56296296296298,2682469.135802469)}
        for name,(conductivity,capacity) in expected.items():
            with self.subTest(material=name):
                self.assertAlmostEqual(material.getfloat(name,'conductivity (w/(m-k))'),conductivity)
                self.assertAlmostEqual(material.getfloat(name,'volumetric heat capacity (j/(m3-k))'),capacity)
        for name in ['material.config','sim.config']:
            self.assertNotIn('${',(self.out/name).read_text())

    def test_checkpoint_tracks_each_runtime_template(self):
        assets = self.custom_assets()
        path, _ = self.save()
        for name, key in [('sim.config.template','atsim_config_sha256'),
                          ('structure.xml.template','atsim_structure_sha256'),
                          ('material.config.template','atsim_material_sha256')]:
            template = assets/name
            original = template.read_bytes()
            template.write_bytes(original+b'\n')
            with self.subTest(template=name),self.assertRaisesRegex(ValueError,key):
                load_model(path,self.params,self.system)
            template.write_bytes(original)

    def test_roundtrip_freezes_weights_but_preserves_input_gradients(self):
        path,_=self.save()
        model=load_model(path,self.params,self.system)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        x=torch.tensor([[5000.,14000.]],requires_grad=True)
        prediction=model((x,x,torch.ones_like(x)*3000,torch.ones_like(x)*3000,torch.ones_like(x)*10))
        gradient=torch.autograd.grad(prediction.sum(),x)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(),0)

    def test_checkpoint_rejects_wrong_boundary_geometry_and_grid(self):
        path,_=self.save()
        for attr,value in [('thermal_convection_resistance',.1),('ambient',25),('num_grid_x',16)]:
            changed=copy.copy(self.params);setattr(changed,attr,value)
            with self.subTest(attr=attr),self.assertRaisesRegex(ValueError,'mismatch'):
                load_model(path,changed,self.system)
        system=copy.copy(self.system);system.powermap=np.array([11.,20.])
        with self.assertRaisesRegex(ValueError,'mismatch'):
            load_model(path,self.params,system)
        with self.assertRaisesRegex(FileNotFoundError,'run_thermal_training.py'):
            load_model(self.out/'missing.pt',self.params,self.system)

    def solver(self):
        solver = ATSimSolver(self.out, self.params)
        solver.set_params(self.system)
        solver.set_pos(self.system.powermap, [np.array([4000.,13000.]), np.array([4000.,10000.])],
                       self.system.node_size_x, self.system.node_size_y)
        return solver

    def raw_chip_cells(self):
        nx, ny = self.params.num_grid_x, self.params.num_grid_y
        w, h = self.system.intp_width*1e-6, self.system.intp_height*1e-6
        bottom, top = chip_layer_bounds()
        return np.array([[i*w/nx, (i+1)*w/nx, j*h/ny, (j+1)*h/ny,
                          bottom, top, 300+i*ny+j]
                         for i in range(nx) for j in range(ny)])

    def test_atsim_config_geometry_and_nonsquare_grid_orientation(self):
        self.params.thermal_convection_resistance = .023
        self.params.thermal_num_threads = 8
        solver = self.solver()
        def fake_wait(*args, **kwargs):
            output = self.out/'outputs/raw'
            output.mkdir(parents=True, exist_ok=True)
            raw = self.raw_chip_cells()
            # Add another layer and reverse rows to detect accidental layer/order assumptions.
            other = raw.copy(); other[:,4:6] = [0., .0002]; other[:,6] = 900.
            np.save(output/'steadytemp.npy', np.concatenate([other, raw[::-1]]))
        process = Mock(returncode=0)
        process.wait.side_effect = fake_wait
        with patch('src.thermal.atsim.subprocess.Popen', return_value=process) as launch:
            field = solver.run()
        np.testing.assert_allclose(field, (np.arange(32)+300).reshape(8,4)-273.15)
        settings = configparser.ConfigParser();settings.read(self.out/'sim.config')
        self.assertAlmostEqual(settings.getfloat('NoPackage','htc_top'),1/(.023*.072**2))
        self.assertEqual(settings.getfloat('NoPackage','htc_bot'),0)
        self.assertEqual(settings.get('Temperature','ambient'),'318.15 K')
        self.assertEqual(settings.getint('Simulation','number_of_core'),8)
        self.assertEqual(settings.getint('MeshConfig','rows'),4)
        self.assertEqual(settings.getint('MeshConfig','cols'),8)
        self.assertEqual(settings.getint('MeshConfig','max_depth'),1)
        for key in ['tiers_active', 'tiers_bulk', 'tiers_chiplet']:
            self.assertEqual(settings.getint('MeshConfig',key),1)
        self.assertEqual(settings.get('Simulation','power_type'),'value')
        package = ET.parse(self.out/'structure.xml').getroot()
        components = package.findall('Component')
        self.assertAlmostEqual(float(components[-2].find('Geometry').get('Length')),.036)
        self.assertAlmostEqual(float(components[-1].find('Geometry').get('Length')),.072)
        self.assertEqual(components[1].get('FillingMaterial'),'Underfill')
        for prefix, material in [('Bump','Ubump'),('Chip','Silicon')]:
            with (self.out/(prefix+'.csv')).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows),2)
            self.assertTrue(all(row['UnitName'].startswith(prefix+'_') and row['Label']==material for row in rows))
            self.assertAlmostEqual(float(rows[0]['X']),.002)
            self.assertAlmostEqual(float(rows[0]['Length']),.004)
        with (self.out/'Chip_power.csv').open() as handle:
            power_rows=list(csv.reader(handle))
        self.assertEqual([float(v) for v in power_rows[1]],[10.,20.])
        self.assertEqual(launch.call_args.kwargs['cwd'],self.out)
        self.assertTrue(launch.call_args.kwargs['start_new_session'])

    def test_atsim_rejects_missing_duplicate_or_invalid_cells(self):
        solver = self.solver()
        raw = self.raw_chip_cells()
        path = self.out/'raw.npy'
        for bad in [raw[:-1], np.repeat(raw[:1],len(raw),axis=0), np.full_like(raw,np.nan), raw[:,:6]]:
            np.save(path,bad)
            with self.assertRaises(ValueError):solver.read_chip_temperature(path)

    def test_chip_midplane_ignores_interposer_temperature(self):
        solver = self.solver()
        raw = self.raw_chip_cells()
        interposer = raw.copy()
        interposer[:,4:6] = [.00027, .00038]
        interposer[:,6] = 800.
        path = self.out/'raw.npy'
        np.save(path, np.concatenate([interposer, raw[::-1]]))
        np.testing.assert_allclose(solver.read_chip_temperature(path),
                                   (np.arange(32)+300).reshape(8,4)-273.15)
        self.assertAlmostEqual(temperature_target()['z_m'],465e-6)
        self.assertEqual(metadata(self.params,self.system)['temperature_target'],temperature_target())

    def test_checkpoint_and_dataset_reject_previous_prediction_plane(self):
        path, _ = self.save()
        saved = torch.load(path,weights_only=True)
        saved['metadata']['temperature_target'] = {'name':'interposer_midplane', 'z_m':325e-6}
        torch.save(saved,path)
        with self.assertRaisesRegex(ValueError,'temperature_target'):
            load_model(path,self.params,self.system)
        with self.assertRaisesRegex(ValueError,'temperature_target'):
            validate_dataset(saved,self.params,self.system)

    def test_atsim_timeout_terminates_worker_group(self):
        solver = self.solver()
        process = Mock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired('ATSim',1),0]
        with patch('src.thermal.atsim.subprocess.Popen',return_value=process), patch('src.thermal.atsim.os.killpg') as kill:
            with self.assertRaisesRegex(RuntimeError,'ATSim failed'):solver.run()
            kill.assert_called_once()
            self.assertEqual(kill.call_args.args[0],12345)

    def test_legacy_solver_checkpoint_is_rejected(self):
        path, _ = self.save()
        saved = torch.load(path, weights_only=True)
        saved['metadata']['format_version'] = 1
        saved['metadata'].pop('solver')
        torch.save(saved,path)
        with self.assertRaisesRegex(ValueError,'mismatch'):
            load_model(path,self.params,self.system)

    def test_dataset_split_training_and_compatibility(self):
        geometry=torch.tensor([[[4000.,4000.,4000.,3000.],[13000.,10000.,5000.,4000.]],[[5000.,5000.,4000.,3000.],[13000.,10000.,5000.,4000.]]])
        power=torch.tensor([[10.,20.],[10.,20.]])
        truth=new_model(self.params,self.system)
        with torch.no_grad():
            truth.amp.fill_(800)
            target=truth(tuple(geometry[:,:,i] for i in range(4))+(power,))
        saved={'metadata':metadata(self.params,self.system),'geometry':geometry,'power':power,
            'temperature_rise':target,'train_samples':1,'test_samples':1}
        report=fit_model(saved,self.params,self.system,self.out/'fitted.pt',self.out)
        self.assertLess(report['final_train_mse'],report['initial_train_mse'])
        changed=copy.copy(self.params);changed.ambient=30
        with self.assertRaisesRegex(ValueError,'mismatch'):
            validate_dataset(saved,changed,self.system)
        bad=dict(saved,temperature_rise=target[:1])
        with self.assertRaisesRegex(ValueError,'tensor'):
            validate_dataset(bad,self.params,self.system)


if __name__ == '__main__':
    unittest.main()
