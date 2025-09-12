"""
LAMMPS File Writer Module.

This module is responsible for writing the LAMMPS data. and in. files based
on the processed GROMACS topology.

Key functionalities:
- Gathers all unique parameters used in the system.
- Creates compact, sequential type IDs for LAMMPS.
- Performs unit conversions (kJ/mol -> kcal/mol, nm -> Angstrom).
- Writes all sections of the LAMMPS data file.
- Writes a complete in. script from a template, allowing for user-defined 
  multi-stage simulation protocols and automatically generating SHAKE commands.
"""
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple
from itertools import combinations

# Unit conversion constants
KCAL_PER_KJ = 1.0 / 4.184
A_PER_NM = 10.0

class LammpsWriter:
    """Writes LAMMPS .data and in. files from the processed topology."""

    def __init__(self, parsed_ff: Dict, system_topology: Dict, gro_atoms, box_dims, **kwargs):
        self.ff = parsed_ff
        self.system = system_topology
        self.gro_atoms = gro_atoms
        self.box_dims = box_dims
        self.global_atom_map = {atom['id']: atom for atom in self.system['atoms']}
        self.type_maps = {}
        self.coeffs = {}
        
        # Store simulation settings from kwargs
        self.protocol = kwargs.get('protocol', 'min-nvt-npt')
        self.thermostat = kwargs.get('thermostat', 'nose-hoover')
        self.barostat = kwargs.get('barostat', 'nose-hoover')
        self.use_shake = kwargs.get('use_shake', False)
        self.use_cm_removal = kwargs.get('use_cm_removal', False)

    def write_lammps_data(self, out_path: str):
        """Writes the main LAMMPS data file."""
        print(f"  -> Writing LAMMPS data file to: {out_path}")
        self._prepare_coefficients()
        
        with open(out_path, 'w') as f:
            self._write_header(f)
            self._write_coeffs(f, 'Masses', 'atom', ['mass'], "{type_id} {mass:.6f} # {name}")
            self._write_coeffs(f, 'Pair Coeffs', 'atom', ['epsilon', 'sigma'], "{type_id} {epsilon:.6f} {sigma:.6f} # {name}")
            self._write_coeffs(f, 'Bond Coeffs', 'bond', ['Kb', 'b0'], "{type_id} {Kb:.4f} {b0:.4f} # {name_str}")
            
            angle_style = self.coeffs['angle']['style']
            if angle_style == 'charmm':
                self._write_coeffs(f, 'Angle Coeffs', 'angle', ['Kth', 'th0', 'Kub', 's0'], "{type_id} {Kth:.4f} {th0:.4f} {Kub:.4f} {s0:.4f} # {name_str}")
            elif angle_style == 'harmonic':
                 self._write_coeffs(f, 'Angle Coeffs', 'angle', ['Kth', 'th0'], "{type_id} {Kth:.4f} {th0:.4f} # {name_str}")
            
            self._write_dihedral_coeffs(f)
            self._write_topology(f)

    def write_lammps_in(self, out_path: str):
        """Writes the LAMMPS input script from a template."""
        print(f"  -> Writing LAMMPS input script to: {out_path}")
        print(f"    -> Generating LAMMPS script for protocol: {self.protocol}")

        fudgeLJ, fudgeQQ = 1.0, 1.0
        if self.ff['globals'].get('defaults'):
            defaults = self.ff['globals']['defaults'][0]
            fudgeLJ, fudgeQQ = float(defaults[3]), float(defaults[4])
        special_bonds_line = f"special_bonds   lj 0.0 0.0 {fudgeLJ} coul 0.0 0.0 {fudgeQQ}"

        angle_style_line = f"angle_style     {self.coeffs['angle']['style']}"
        dihedral_style_line = f"dihedral_style  {self.coeffs['dihedral']['style']}"
        
        # Generate optional fixes and the main simulation script
        shake_cmd, cm_removal_cmd = self._generate_optional_fixes()
        simulation_script = self._generate_simulation_script()

        template_path = Path(__file__).parent / 'lammps_template.in'
        with open(template_path, 'r') as fin:
            template = fin.read()

        text = template.replace('{data_filename}', Path(out_path).name[3:] + '.data')
        text = text.replace('{special_bonds_line}', special_bonds_line)
        text = text.replace('{angle_style_line}', angle_style_line)
        text = text.replace('{dihedral_style_line}', dihedral_style_line)
        text = text.replace('{shake_command}', shake_cmd)
        text = text.replace('{cm_removal_command}', cm_removal_cmd)
        text = text.replace('{simulation_script}', simulation_script)
        text = text.replace('{trajectory_basename}', Path(out_path).name[3:])

        with open(out_path, 'w') as f:
            f.write(text)

    def _generate_optional_fixes(self) -> Tuple[str, str]:
        """Generates command blocks for optional fixes like SHAKE and CM-removal."""
        # --- SHAKE Command Generation ---
        shake_cmd = ""
        if self.use_shake:
            h_types = {p[0] for p in self.ff['globals']['atomtypes'] if float(p[2]) < 1.1}
            if h_types:
                h_bond_type_ids = [str(tid) for bt, tid in self.type_maps['bond'].items() if bt[0] in h_types or bt[1] in h_types]
                if h_bond_type_ids:
                    bond_list = " ".join(sorted(h_bond_type_ids, key=int))
                    shake_cmd = f"fix               shake all shake 0.0001 20 0 b {bond_list}"
                    print(f"    -> Identified {len(h_bond_type_ids)} bond types with Hydrogen for SHAKE.")
                else: print("    -> WARNING: --shake requested, but no H-bonds found in the system.")
            else: print("    -> WARNING: --shake requested, but no hydrogen atom types found in force field.")
        
        # --- Center-of-Mass Removal Command ---
        cm_removal_cmd = ""
        if self.use_cm_removal:
            cm_removal_cmd = "fix               mom_removal all momentum 100 linear 1 1 1"
        
        return shake_cmd, cm_removal_cmd
    
    def _generate_simulation_script(self) -> str:
        """Generates the full multi-stage simulation script based on the protocol."""
        script_blocks = []
        steps = self.protocol.lower().split('-')
        
        # Validate protocol steps
        valid_steps = {'min', 'nvt', 'npt'}
        if any(step not in valid_steps for step in steps):
            raise ValueError(f"Invalid protocol step found. Allowed steps are: {list(valid_steps)}")

        if 'min' in steps:
            script_blocks.append(
                "# --- 1. Minimization ---\n"
                'print "--- Starting Energy Minimization ---"\n'
                "minimize        1.0e-4 1.0e-6 1000 10000\n"
                'print "--- Minimization Complete ---"'
            )

        if 'nvt' in steps:
            nvt_lines = [
                "\n# --- 2. NVT Equilibration ---",
                'print "--- Starting NVT Equilibration ---"',
                "reset_timestep  0",
                "velocity        all create 300.0 4928459 rot yes dist gaussian",
            ]
            if self.thermostat == 'nose-hoover':
                nvt_lines.append("fix             1 all nvt temp 300.0 300.0 $(100.0*dt)")
            elif self.thermostat == 'langevin':
                nvt_lines.append("fix             1 all langevin 300.0 300.0 100.0 48279")
            elif self.thermostat == 'berendsen':
                nvt_lines.append("fix             1 all temp/berendsen 300.0 300.0 100.0")
                nvt_lines.append("fix             2 all nve")
            
            nvt_lines.extend([
                "dump            nvt_dcd all dcd 1000 {trajectory_basename}_nvt.dcd",
                "dump_modify     nvt_dcd unwrap yes",
                "run             50000  # 100 ps",
                "unfix           1",
            ])
            if self.thermostat == 'berendsen': nvt_lines.append("unfix           2")
            nvt_lines.extend(["undump          nvt_dcd", 'print "--- NVT Equilibration Complete ---"'])
            script_blocks.append("\n".join(nvt_lines))

        if 'npt' in steps:
            npt_lines = [
                "\n# --- 3. NPT Production ---",
                'print "--- Starting NPT Production ---"',
                "reset_timestep  0",
            ]
            if self.barostat == 'nose-hoover':
                 npt_lines.append("# fix             1 all npt temp 300.0 300.0 $(100.0*dt) iso 1.0 1.0 $(1000.0*dt)")
                 npt_lines.append("fix             1 all npt temp 300.0 300.0 $(100.0*dt) x 1.0 1.0 $(100.0*dt) y 1.0 1.0 $(100.0*dt) z 1.0 1.0 $(100.0*dt) couple xy")
            elif self.barostat == 'berendsen':
                 npt_lines.append("fix             1 all press/berendsen iso 1.0 1.0 1000.0")
                 npt_lines.append("fix             2 all nvt temp 300.0 300.0 $(100.0*dt)") # Berendsen P needs separate T

            npt_lines.extend([
                "dump            npt_dcd all dcd 100 {trajectory_basename}_npt.dcd",
                "dump_modify     npt_dcd unwrap yes",
                "run             250000 # 500 ps",
                'print "--- NPT Production Complete ---"',
            ])
            script_blocks.append("\n".join(npt_lines))
        
        return "\n\n".join(script_blocks)


    # ... (the rest of the class is unchanged) ...
    def _prepare_coefficients(self):
        """Gathers all unique parameters and creates mappings to LAMMPS type IDs."""
        print("    -> Filtering force field parameters used by the system.")
        all_atom_types = self.ff['globals']['atomtypes']
        atom_params = {p[0]: {'mass': float(p[2]), 'sigma': float(p[5]) * A_PER_NM, 'epsilon': float(p[6]) * KCAL_PER_KJ} for p in all_atom_types}
        used_atom_types = sorted({atom['type'] for atom in self.system['atoms']})
        self.type_maps['atom'] = {name: i + 1 for i, name in enumerate(used_atom_types)}
        self.coeffs['atom'] = {name: atom_params[name] for name in used_atom_types}

        all_bond_types = self.ff['globals']['bondtypes']
        bond_params = {tuple(sorted((p[0], p[1]))): {'b0': float(p[3]) * A_PER_NM, 'Kb': float(p[4]) * KCAL_PER_KJ / (A_PER_NM**2)} for p in all_bond_types}
        used_bond_types = sorted({tuple(sorted((self.global_atom_map[b[0]]['type'], self.global_atom_map[b[1]]['type']))) for b in self.system['bonds']})
        self.type_maps['bond'] = {name: i + 1 for i, name in enumerate(used_bond_types)}
        self.coeffs['bond'] = {name: bond_params[name] for name in used_bond_types}
        
        all_angle_types = self.ff['globals']['angletypes']
        angle_params = {}
        angle_func_types = []
        for p in all_angle_types:
            key = (p[0], p[1], p[2])
            func = int(p[3])
            angle_func_types.append(func)
            
            if func == 5:
                kub = float(p[7]) * KCAL_PER_KJ / (A_PER_NM**2) if len(p) > 7 else 0.0
                s0 = float(p[6]) * A_PER_NM if len(p) > 6 else 0.0
                angle_params[key] = {'th0': float(p[4]), 'Kth': float(p[5]) * KCAL_PER_KJ, 'Kub': kub, 's0': s0}
            elif func == 1:
                angle_params[key] = {'th0': float(p[4]), 'Kth': float(p[5]) * KCAL_PER_KJ}

        func_counts = Counter(angle_func_types)
        dominant_func = func_counts.most_common(1)[0][0] if func_counts else 5
        style_map = {1: 'harmonic', 5: 'charmm'}
        lammps_style = style_map.get(dominant_func, 'charmm')
        print(f"    -> Detected GROMACS angle function type {dominant_func}, setting LAMMPS style to '{lammps_style}'.")

        used_angle_types = sorted({self._find_parameter_key(a, angle_params, 3) for a in self.system['angles']})
        self.type_maps['angle'] = {name: i + 1 for i, name in enumerate(used_angle_types)}
        self.coeffs['angle'] = {'style': lammps_style, 'params': {name: angle_params[name] for name in used_angle_types}}
        
        all_dihedral_params = self.ff['globals']['processed_dihedrals']
        used_dihedral_keys = {self._find_parameter_key(d, all_dihedral_params, 4) for d in self.system['dihedrals']}
        used_dihedrals = {key: all_dihedral_params[key] for key in used_dihedral_keys}
        
        func_counts = Counter(term['func'] for params in used_dihedrals.values() for term in params)
        dominant_func = func_counts.most_common(1)[0][0] if func_counts else 9
        style_map = {9: 'fourier', 3: 'charmm', 4: 'charmm', 1: 'multi/harmonic'}
        lammps_style = style_map.get(dominant_func, 'fourier')
        print(f"    -> Detected GROMACS dihedral function type {dominant_func}, setting LAMMPS style to '{lammps_style}'.")
        
        sorted_keys = sorted(list(used_dihedrals.keys()))
        self.type_maps['dihedral'] = {name: i + 1 for i, name in enumerate(sorted_keys)}
        self.coeffs['dihedral'] = {'style': lammps_style, 'params': used_dihedrals}

    def _generate_patterns(self, types: List[str]) -> List[tuple]:
        num_types = len(types)
        patterns = []
        for num_wildcards in range(num_types + 1):
            for positions in combinations(range(num_types), num_wildcards):
                pattern = list(types)
                for pos in positions:
                    pattern[pos] = 'X'
                patterns.append(tuple(pattern))
        return patterns

    def _find_parameter_key(self, atoms: List[int], param_dict: Dict, num_atoms: int) -> tuple:
        types = [self.global_atom_map[i]['type'] for i in atoms[:num_atoms]]
        forward_patterns = self._generate_patterns(types)
        reversed_patterns = self._generate_patterns(types[::-1])
        patterns_to_check = list(dict.fromkeys(forward_patterns + reversed_patterns))
        for pattern in patterns_to_check:
            if pattern in param_dict:
                return pattern
        raise ValueError(f"No matching parameter found for types: {'-'.join(types)}")

    def _write_header(self, f):
        f.write("LAMMPS data file (converted from GROMACS)\n\n")
        f.write(f"{len(self.system['atoms'])} atoms\n")
        f.write(f"{len(self.system['bonds'])} bonds\n")
        f.write(f"{len(self.system['angles'])} angles\n")
        f.write(f"{len(self.system['dihedrals'])} dihedrals\n\n")
        f.write(f"{len(self.type_maps['atom'])} atom types\n")
        f.write(f"{len(self.type_maps['bond'])} bond types\n")
        f.write(f"{len(self.type_maps['angle'])} angle types\n")
        f.write(f"{len(self.type_maps['dihedral'])} dihedral types\n\n")
        f.write(f"0.0 {self.box_dims[0]:.6f} xlo xhi\n")
        f.write(f"0.0 {self.box_dims[1]:.6f} ylo yhi\n")
        f.write(f"0.0 {self.box_dims[2]:.6f} zlo zhi\n")

    def _write_coeffs(self, f, title, coeff_type, param_keys, fmt_str):
        f.write(f"\n{title}\n\n")
        
        param_source = self.coeffs[coeff_type]
        if 'params' in param_source:
             param_source = param_source['params']

        for name, type_id in self.type_maps[coeff_type].items():
            params = param_source[name]
            fmt_dict = {'type_id': type_id, 'name': name}
            fmt_dict.update({key: params.get(key, 0.0) for key in param_keys})
            if isinstance(name, tuple):
                fmt_dict['name_str'] = '-'.join(name)
            f.write(fmt_str.format(**fmt_dict) + "\n")

    def _write_dihedral_coeffs(self, f):
        f.write("\nDihedral Coeffs\n\n")
        style = self.coeffs['dihedral']['style']
        
        for name, type_id in self.type_maps['dihedral'].items():
            terms = self.coeffs['dihedral']['params'][name]
            line_parts = [str(type_id)]
            
            if style == 'fourier':
                line_parts.append(str(len(terms)))
                for term in terms:
                    k = float(term['params'][1]) * KCAL_PER_KJ
                    n = int(float(term['params'][2]))
                    d = 180.0 if float(term['params'][0]) == 0 else 0.0
                    line_parts.extend([f"{k:.4f}", f"{n}", f"{d:.0f}"])
            elif style == 'charmm':
                term = terms[0]
                k = float(term['params'][1]) * KCAL_PER_KJ
                n = int(float(term['params'][2]))
                d = float(term['params'][0])
                w = 1.0
                line_parts.extend([f"{k:.4f}", f"{n}", f"{int(d)}", f"{w:.1f}"])

            f.write(" ".join(line_parts) + f" # {'-'.join(name)}\n")

    def _write_topology(self, f):
        f.write("\nAtoms\n\n")
        for i, atom_coords in enumerate(self.gro_atoms.positions):
            atom_info = self.global_atom_map[i + 1]
            type_id = self.type_maps['atom'][atom_info['type']]
            f.write(f"{atom_info['id']} {atom_info['mol_id']} {type_id} {atom_info['charge']:.4f} "
                    f"{atom_coords[0]:.4f} {atom_coords[1]:.4f} {atom_coords[2]:.4f}\n")

        f.write("\nBonds\n\n")
        for i, b in enumerate(self.system['bonds']):
            key = tuple(sorted((self.global_atom_map[b[0]]['type'], self.global_atom_map[b[1]]['type'])))
            type_id = self.type_maps['bond'][key]
            f.write(f"{i + 1} {type_id} {b[0]} {b[1]}\n")

        f.write("\nAngles\n\n")
        for i, a in enumerate(self.system['angles']):
            key = self._find_parameter_key(a, self.coeffs['angle']['params'], 3)
            type_id = self.type_maps['angle'][key]
            f.write(f"{i + 1} {type_id} {a[0]} {a[1]} {a[2]}\n")
            
        f.write("\nDihedrals\n\n")
        for i, d in enumerate(self.system['dihedrals']):
            key = self._find_parameter_key(d, self.coeffs['dihedral']['params'], 4)
            type_id = self.type_maps['dihedral'][key]
            f.write(f"{i + 1} {type_id} {d[0]} {d[1]} {d[2]} {d[3]}\n")