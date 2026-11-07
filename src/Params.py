"""Grouped JSON configuration with flat attributes for the placement kernels."""
import json
from copy import deepcopy
from pathlib import Path


class Params:
    _RENAMED_OPTIONS = {
        'ILPsolver': 'ilp_solver',
        'training_ILPsolver': 'training_ilp_solver',
        'RePlAce_ref_hpwl': 'ref_hpwl',
        'OVLP_LOWER_PCOF': 'density_weight_lower_ratio',
        'OVLP_UPPER_PCOF': 'density_weight_upper_ratio',
        'Llambda_density_weight_iteration': 'density_weight_iterations',
        'Lsub_iteration': 'sub_iterations',
        'pertub_flag': 'perturb_flag',
    }

    @staticmethod
    def _runtime_key(section, key):
        # Keep identically named public settings independent in the flat runtime API.
        return 'thermal_num_threads' if section == 'thermal' and key == 'num_threads' else key

    @staticmethod
    def _config_key(key):
        return 'num_threads' if key == 'thermal_num_threads' else key

    # These options are fixed or have no implemented effect in the current flow.
    _REMOVED_OPTIONS = frozenset({
        'evaluate_thermal', 'temp_aware_opt', 'flip_opt', 'warp_aware_opt', 'warp_weight_init', 'warp_threshold',
        'reso_interposer', 'gpu', 'target_density', 'wirelength', 'ignore_net_degree',
        'random_center_init_flag', 'floorplan_stages',
    })

    def __init__(self, filename=None):
        if filename is None or not Path(filename).exists():
            filename = Path(__file__).resolve().parents[1] / 'params.json'
        definitions = json.loads(Path(filename).read_text())
        self.params_dict = {}
        self._sections = {}
        groups = definitions if 'opt' in definitions or 'thermal' in definitions else {'opt': definitions}
        for section, fields in groups.items():
            if section not in ('opt', 'thermal') or not isinstance(fields, dict):
                raise ValueError('Parameter sections must be opt or thermal objects')
            for key, definition in fields.items():
                if key == 'description' and isinstance(definition, str):
                    continue  # Section metadata, not a runtime parameter.
                if key in self._REMOVED_OPTIONS:
                    raise ValueError(f'{key} is no longer configurable; remove it from the parameter file')
                key = self._runtime_key(section, key)
                if key in self.params_dict:
                    raise ValueError(f'Duplicate parameter: {key}')
                self.params_dict[key] = deepcopy(definition)
                self._sections[key] = section
                setattr(self, key, deepcopy(definition.get('default')))

    @staticmethod
    def _merge(current, override):
        if isinstance(current, dict) and isinstance(override, dict):
            result = deepcopy(current)
            for key, value in override.items():
                result[key] = Params._merge(result.get(key), value)
            return result
        return deepcopy(override)

    def fromJson(self, data):
        # Flat overrides remain readable for historical parameter files.
        grouped = 'opt' in data or 'thermal' in data
        groups = data if grouped else {None: data}
        seen = set()
        for section, fields in groups.items():
            if section not in (None, 'opt', 'thermal') or not isinstance(fields, dict):
                raise ValueError('Parameter sections must be opt or thermal objects')
            for key, value in fields.items():
                if section is not None and key == 'description' and isinstance(value, str):
                    continue
                if key in self._REMOVED_OPTIONS:
                    raise ValueError(f'{key} is no longer configurable; remove it from the parameter file')
                key = self._RENAMED_OPTIONS.get(key, key)
                key = self._runtime_key(section, key)
                if key in seen:
                    raise ValueError(f'Duplicate parameter: {key}')
                seen.add(key)
                if key in self._sections and section is not None and self._sections[key] != section:
                    raise ValueError(f'{key} belongs in the {self._sections[key]} section')
                self._sections.setdefault(key, section or 'opt')
                setattr(self, key, self._merge(getattr(self, key, None), value))

    def toJson(self):
        result = {'opt': {}, 'thermal': {}}
        for key, value in self.__dict__.items():
            if key in ('params_dict', 'floorplan_stages', 'temp_aware_opt') or key.startswith('_'):
                continue
            result[self._sections.get(key, 'opt')][self._config_key(key)] = deepcopy(value)
        return result

    def toMarkdownTable(self):
        lines = ['| Parameter | Default | Description |', '| --- | --- | --- |']
        for key, definition in self.params_dict.items():
            lines.append(f"| {self._sections[key]}.{self._config_key(key)} | {definition.get('default')} | {definition.get('description', '')} |")
        return '\n'.join(lines)+'\n'

    def printHelp(self):
        print(self.toMarkdownTable())

    def dump(self, filename):
        Path(filename).write_text(json.dumps(self.toJson(), indent=2)+'\n')

    def load(self, filename):
        self.fromJson(json.loads(Path(filename).read_text()))

    def __str__(self):
        return str(self.toJson())

    def __repr__(self):
        return self.__str__()
