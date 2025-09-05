"""
LAMMPS File Writer Module.

This module is responsible for writing the LAMMPS .data and .in files based
on the processed GROMACS topology.

Key functionalities:
- Gathers all unique parameters used in the system.
- Creates compact, sequential type IDs for LAMMPS.
- Performs unit conversions (kJ/mol -> kcal/mol, nm -> Angstrom).
- Writes all sections of the LAMMPS data file (Header, Masses, Coeffs, Topology).
- Determines the appropriate dihedral style and writes a complete .in script from a template.
"""
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List
from itertools import combinations

# Unit conversion constants
KCAL_PER_KJ = 1.0 / 4.184
A_PER_NM = 10.0

class LammpsWriter:
    """Writes LAMMPS .data and .in files from the processed topology."""

    def __init__(self, parsed_ff: Dict, system_topology: Dict, gro_atoms, box_dims):
        self.ff = parsed_ff
        self.system = system_topology
        self.gro_atoms = gro_atoms
        self.box_dims = box_dims
        self.global_atom_map = {atom['id']: atom for atom in self.system['atoms']}
        self.type_maps = {}
        self.coeffs = {}

    def write_lammps_data(self, out_path: str):
        """Writes the main LAMMPS data file."""
        print(f"  -> Writing LAMMPS data file to: {out_path}")
        self._prepare_coefficients()
        
        with open(out_path, 'w') as f:
            self._write_header(f)
            self._write_coeffs(f, 'Masses', 'atom', ['mass'], "{type_id} {mass:.6f} # {name}")
            self._write_coeffs(f, 'Pair Coeffs', 'atom', ['epsilon', 'sigma'], "{type_id} {epsilon:.6f} {sigma:.6f} # {name}")
            self._write_coeffs(f, 'Bond Coeffs', 'bond', ['Kb', 'b0'], "{type_id} {Kb:.4f} {b0:.4f} # {name_str}")
            self._write_coeffs(f, 'Angle Coeffs', 'angle', ['Kth', 'th0'], "{type_id} {Kth:.4f} {th0:.4f} # {name_str}")
            self._write_dihedral_coeffs(f)
            self._write_topology(f)

    def write_lammps_in(self, out_path: str):
        """Writes the LAMMPS input script from a template."""
        print(f"  -> Writing LAMMPS input script to: {out_path}")

        # Determine special bonds
        fudgeLJ, fudgeQQ = 1.0, 1.0
        if self.ff['globals'].get('defaults'):
            defaults = self.ff['globals']['defaults'][0]
            fudgeLJ, fudgeQQ = float(defaults[3]), float(defaults[4])
        special_bonds_line = f"special_bonds   lj 0.0 0.0 {fudgeLJ} coul 0.0 0.0 {fudgeQQ}"

        # Determine dihedral style
        dihedral_style_line = f"dihedral_style  {self.coeffs['dihedral']['style']}"

        template_path = Path(__file__).parent / 'lammps_template.in'
        with open(template_path, 'r') as fin:
            template = fin.read()

        text = template.replace('{data_filename}', Path(out_path).stem + '.data')
        text = text.replace('{trajectory_basename}', Path(out_path).stem)
        text = text.replace('{special_bonds_line}', special_bonds_line)
        text = text.replace('{dihedral_style_line}', dihedral_style_line)

        with open(out_path, 'w') as f:
            f.write(text)
    
    def _prepare_coefficients(self):
        """Gathers all unique parameters and creates mappings to LAMMPS type IDs."""
        print("    -> Filtering force field parameters used by the system.")
        # Atoms
        all_atom_types = self.ff['globals']['atomtypes']
        atom_params = {p[0]: {'mass': float(p[2]), 'sigma': float(p[5]) * A_PER_NM, 'epsilon': float(p[6]) * KCAL_PER_KJ} for p in all_atom_types}
        used_atom_types = sorted({atom['type'] for atom in self.system['atoms']})
        self.type_maps['atom'] = {name: i + 1 for i, name in enumerate(used_atom_types)}
        self.coeffs['atom'] = {name: atom_params[name] for name in used_atom_types}

        # Bonds
        all_bond_types = self.ff['globals']['bondtypes']
        bond_params = {tuple(sorted((p[0], p[1]))): {'b0': float(p[3]) * A_PER_NM, 'Kb': float(p[4]) * KCAL_PER_KJ / (A_PER_NM**2)} for p in all_bond_types}
        used_bond_types = sorted({tuple(sorted((self.global_atom_map[b[0]]['type'], self.global_atom_map[b[1]]['type']))) for b in self.system['bonds']})
        self.type_maps['bond'] = {name: i + 1 for i, name in enumerate(used_bond_types)}
        self.coeffs['bond'] = {name: bond_params[name] for name in used_bond_types}
        
        # Angles
        all_angle_types = self.ff['globals']['angletypes']
        angle_params = {(p[0], p[1], p[2]): {'th0': float(p[4]), 'Kth': float(p[5]) * KCAL_PER_KJ} for p in all_angle_types}
        used_angle_types = sorted({self._find_parameter_key(a, angle_params, 3) for a in self.system['angles']})
        self.type_maps['angle'] = {name: i + 1 for i, name in enumerate(used_angle_types)}
        self.coeffs['angle'] = {name: angle_params[name] for name in used_angle_types}
        
        # Dihedrals
        all_dihedral_params = self.ff['globals']['processed_dihedrals']
        used_dihedral_keys = {self._find_parameter_key(d, all_dihedral_params, 4) for d in self.system['dihedrals']}
        used_dihedrals = {key: all_dihedral_params[key] for key in used_dihedral_keys}
        
        # Determine dominant dihedral style
        func_counts = Counter(term['func'] for params in used_dihedrals.values() for term in params)
        dominant_func = func_counts.most_common(1)[0][0] if func_counts else 9
        
        style_map = {9: 'fourier', 3: 'charmm', 4: 'charmm', 1: 'multi/harmonic'}
        lammps_style = style_map.get(dominant_func, 'fourier')
        print(f"    -> Detected GROMACS dihedral function type {dominant_func}, setting LAMMPS style to '{lammps_style}'.")
        
        sorted_keys = sorted(list(used_dihedrals.keys()))
        self.type_maps['dihedral'] = {name: i + 1 for i, name in enumerate(sorted_keys)}
        self.coeffs['dihedral'] = {'style': lammps_style, 'params': used_dihedrals}


    def _generate_patterns(self, types: List[str]) -> List[tuple]:
        """
        Generates a list of patterns from most specific to most general by
        iteratively replacing atom types with wildcards ('X').
        """
        num_types = len(types)
        patterns = []
        
        for num_wildcards in range(num_types + 1):
            wildcard_positions_iter = combinations(range(num_types), num_wildcards)
            for positions in wildcard_positions_iter:
                pattern = list(types)
                for pos in positions:
                    pattern[pos] = 'X'
                patterns.append(tuple(pattern))
                
        return patterns

    def _find_parameter_key(self, atoms: List[int], param_dict: Dict, num_atoms: int) -> tuple:
        """
        Finds the best-matching parameter key by checking a prioritized list of
        patterns with wildcards, for both forward and reversed atom types.
        """
        types = [self.global_atom_map[i]['type'] for i in atoms[:num_atoms]]
        
        forward_patterns = self._generate_patterns(types)
        reversed_patterns = self._generate_patterns(types[::-1])

        # Combine and prioritize. Using dict.fromkeys to remove duplicates 
        # while preserving the order of appearance.
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
        for name, type_id in self.type_maps[coeff_type].items():
            params = self.coeffs[coeff_type][name]
            fmt_dict = {'type_id': type_id, 'name': name}
            fmt_dict.update({key: params[key] for key in param_keys})
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
                    # GMX funct 9: phi0, k, n -> LAMMPS fourier: K, n, d
                    k = float(term['params'][1]) * KCAL_PER_KJ
                    n = int(float(term['params'][2]))
                    d = 180.0 if float(term['params'][0]) == 0 else 0.0 # Heuristic check
                    line_parts.extend([f"{k:.4f}", f"{n}", f"{d:.0f}"])
            elif style == 'charmm':
                # GMX funct 3/4: phi0, k, n -> LAMMPS charmm: K, n, d, w
                term = terms[0] # CHARMM style is not multi-term in LAMMPS
                k = float(term['params'][1]) * KCAL_PER_KJ
                n = int(float(term['params'][2]))
                d = float(term['params'][0])
                w = 1.0 # Default weight
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
            key = self._find_parameter_key(a, self.coeffs['angle'], 3)
            type_id = self.type_maps['angle'][key]
            f.write(f"{i + 1} {type_id} {a[0]} {a[1]} {a[2]}\n")
            
        f.write("\nDihedrals\n\n")
        for i, d in enumerate(self.system['dihedrals']):
            key = self._find_parameter_key(d, self.coeffs['dihedral']['params'], 4)
            type_id = self.type_maps['dihedral'][key]
            f.write(f"{i + 1} {type_id} {d[0]} {d[1]} {d[2]} {d[3]}\n")
