import os
import re
import argparse
import MDAnalysis as mda
from os.path import dirname
from collections import defaultdict
from typing import Optional
from pathlib import Path

# --- GROMACS Parser Functions ---

def _parse_gromacs_topology(filepath, data_struct, visited_files):
    """
    Recursively parses a GROMACS topology file (.top or .itp), correctly
    scoping parameters within [ moleculetype ] blocks.
    Handles sections, #include, and #ifdef/#endif directives.
    """
    abs_path = os.path.abspath(filepath)
    if abs_path in visited_files:
        return
    
    print(f"--- Parsing file: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Topology file not found: {filepath}")

    visited_files.add(abs_path)
    current_file_dir = os.path.dirname(abs_path)
    
    current_section = None
    current_mol_type = None
    ifdef_level = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.split(';')[0].strip()
            if not line:
                continue

            if line.startswith('#ifdef'):
                ifdef_level += 1
                continue
            elif line.startswith('#endif'):
                if ifdef_level > 0: ifdef_level -= 1
                continue
            
            if ifdef_level > 0:
                continue
            
            if line.startswith('#include'):
                rel_path = line.split('"')[1]
                include_path = os.path.join(current_file_dir, rel_path)
                _parse_gromacs_topology(include_path, data_struct, visited_files)
                continue

            match = re.match(r'\[\s*(\w+)\s*\]', line)
            if match:
                current_section = match.group(1).lower()
                if current_section == 'moleculetype':
                    current_mol_type = None # Reset on new moleculetype section
                continue

            if current_section:
                if current_section == 'moleculetype':
                    if not current_mol_type:
                        current_mol_type = line.split()[0]
                        if current_mol_type not in data_struct['molecules']:
                            data_struct['molecules'][current_mol_type] = defaultdict(list)
                elif current_section in ['system', 'molecules']:
                    data_struct['system'][current_section].append(line.split())
                elif current_mol_type:
                    data_struct['molecules'][current_mol_type][current_section].append(line.split())
                else:
                    data_struct['globals'][current_section].append(line.split())

def parse_top_level_topology(top_path):
    """
    Entry point for parsing. Initializes the data structure and starts the recursion.
    """
    print(f"--- Starting topology parsing from: {top_path}")
    data_struct = {
        "globals": defaultdict(list),
        "molecules": {},
        "system": defaultdict(list)
    }
    visited_files = set()
    _parse_gromacs_topology(top_path, data_struct, visited_files)
    return data_struct

def parse_gro_mda(gro_path):
    """Parses a .gro file using MDAnalysis."""
    print(f"--- Parsing GRO file with MDAnalysis: {gro_path}")
    if not os.path.exists(gro_path):
        raise FileNotFoundError(f"GRO file not found: {gro_path}")
    universe = mda.Universe(gro_path)
    box_dims = universe.dimensions[:3]
    return universe.atoms, box_dims

# --- Parameter Matching Functions ---

def find_angle_type_key(t1, t2, t3, all_angle_types):
    """Finds matching angle type key, considering symmetry and wildcards."""
    if t1=='OT' or t2=='OT' or t3=='OT':
        pass
    patterns = [(t1, t2, t3), (t3, t2, t1), ('X', t2, t3), (t3, t2, 'X'), (t1, 'X', t3), (t3, 'X', t1), (t1, t2, 'X'), ('X', t2, t1), ('X', 'X', t3), (t3, 'X', 'X'), ('X', t2, 'X'), (t1, 'X', 'X'), ('X', 'X', t1), ('X', 'X', 'X')]
    for p in patterns:
        if p in all_angle_types: return p
    raise ValueError(f"No matching angle parameter for types: {t1}-{t2}-{t3}")

def find_dihedral_type_key(t1, t2, t3, t4, all_dihedral_types):
    """Finds matching dihedral type key, considering symmetry and wildcards."""
    patterns = [(t1, t2, t3, t4), (t4, t3, t2, t1), ('X', t2, t3, t4), (t4, t3, t2, 'X'), (t1, t2, t3, 'X'), ('X', t3, t2, t1), (t1, 'X', t3, t4), (t4, t3, 'X', t1), (t1, t2, 'X', t4), (t4, 'X', t2, t1), ('X', t2, t3, 'X'), ('X', t3, t2, 'X'), ('X', 'X', t3, t4), (t4, t3, 'X', 'X'), (t1, 'X', 'X', t4), (t4, 'X', 'X', t1), (t1, t2, 'X', 'X'), ('X', 'X', t2, t1), ('X', t2, 'X', t4), (t4, 'X', t2, 'X'), ('X', 'X', 'X', t4), (t4, 'X', 'X', 'X'), ('X', 'X', t3, 'X'), ('X', t3, 'X', 'X'), ('X', t2, 'X', 'X'), ('X', 'X', t2, 'X'), (t1, 'X', 'X', 'X'), ('X', 'X', 'X', t1), ('X', 'X', 'X', 'X')]
    for p in patterns:
        if p in all_dihedral_types: return p
    raise ValueError(f"No matching dihedral parameter for types: {t1}-{t2}-{t3}-{t4}")

# --- LAMMPS Writer Functions ---

def write_lammps_data(ff_coeffs, system_topology, gro_atoms_mda, box_dims, data_filename="lammps.data"):
    """Writes the parsed and processed data to a LAMMPS data file."""
    print(f"--- Filtering and writing LAMMPS data file: {data_filename}")

    KCAL_PER_KJ = 1.0 / 4.184
    A_PER_NM    = 10
    global_atom_map = {atom['id']: atom for atom in system_topology['atoms']}

    # --- Step 1: Identify all unique parameter types *used* by the system ---
    used_atom_types = {atom['type'] for atom in system_topology['atoms']}
    
    used_bond_types = set()
    for b in system_topology['bonds']:
        t1 = global_atom_map[b[0]]['type']
        t2 = global_atom_map[b[1]]['type']
        used_bond_types.add(tuple(sorted((t1, t2))))

    used_angle_types = set()
    for a in system_topology['angles']:
        t1 = global_atom_map[a[0]]['type']
        t2 = global_atom_map[a[1]]['type']
        t3 = global_atom_map[a[2]]['type']
        used_angle_types.add(find_angle_type_key(t1, t2, t3, ff_coeffs['angletypes']))

    used_dihedral_types = set()
    for d in system_topology['dihedrals']:
        t1 = global_atom_map[d[0]]['type']
        t2 = global_atom_map[d[1]]['type']
        t3 = global_atom_map[d[2]]['type']
        t4 = global_atom_map[d[3]]['type']
        used_dihedral_types.add(find_dihedral_type_key(t1, t2, t3, t4, ff_coeffs['dihedraltypes']))
    
    print(f"--- Found {len(used_atom_types)} unique atom types in the system.")
    print(f"--- Found {len(used_bond_types)} unique bond types in the system.")
    print(f"--- Found {len(used_angle_types)} unique angle types in the system.")
    print(f"--- Found {len(used_dihedral_types)} unique dihedral types in the system.")

    # --- Step 2: Create filtered coefficient dicts and new, compact type maps ---
    filtered_atom_coeffs = {key: ff_coeffs['atomtypes'][key] for key in used_atom_types}
    atom_type_map = {name: i + 1 for i, name in enumerate(filtered_atom_coeffs)}

    filtered_bond_coeffs = {key: ff_coeffs['bondtypes'][key] for key in used_bond_types}
    bond_type_map = {name: i + 1 for i, name in enumerate(filtered_bond_coeffs)}

    filtered_angle_coeffs = {key: ff_coeffs['angletypes'][key] for key in used_angle_types}
    angle_type_map = {name: i + 1 for i, name in enumerate(filtered_angle_coeffs)}

    filtered_dihedral_coeffs = {key: ff_coeffs['dihedraltypes'][key] for key in used_dihedral_types}
    dihedral_type_map = {name: i + 1 for i, name in enumerate(filtered_dihedral_coeffs)}

    with open(data_filename, 'w') as f:
        # --- Header (using lengths of filtered maps) ---
        f.write("LAMMPS data file (converted from GROMACS)\n\n")
        f.write(f"{len(system_topology['atoms'])} atoms\n")
        f.write(f"{len(system_topology['bonds'])} bonds\n")
        f.write(f"{len(system_topology['angles'])} angles\n")
        f.write(f"{len(system_topology['dihedrals'])} dihedrals\n\n")
        f.write(f"{len(atom_type_map)} atom types\n")
        f.write(f"{len(bond_type_map)} bond types\n")
        f.write(f"{len(angle_type_map)} angle types\n")
        f.write(f"{len(dihedral_type_map)} dihedral types\n\n")
        f.write(f"0.0 {box_dims[0]:.6f} xlo xhi\n0.0 {box_dims[1]:.6f} ylo yhi\n0.0 {box_dims[2]:.6f} zlo zhi\n")

        # --- Coefficients (writing from filtered dicts) ---
        f.write("\nMasses\n\n")
        for name, type_id in atom_type_map.items(): 
            f.write(f"{type_id} {filtered_atom_coeffs[name]['mass']} # {name}\n")
            
        f.write("\nPair Coeffs\n\n")
        for name, type_id in atom_type_map.items():
            p = filtered_atom_coeffs[name]
            f.write(f"{type_id} {float(p['epsilon'])*KCAL_PER_KJ:.6f} {float(p['sigma'])*A_PER_NM:.6f} # {name}\n")

        f.write("\nBond Coeffs\n\n")
        for name, type_id in bond_type_map.items():
            p = filtered_bond_coeffs[name]
            f.write(f"{type_id} {float(p['Kb'])*KCAL_PER_KJ/A_PER_NM**2:.4f} {float(p['b0'])*A_PER_NM:.4f} # {name[0]}-{name[1]}\n")

        f.write("\nAngle Coeffs\n\n")
        for name, type_id in angle_type_map.items():
            p = filtered_angle_coeffs[name]
            f.write(f"{type_id} {float(p['Kth'])*KCAL_PER_KJ:.4f} {float(p['th0']):.4f} {float(p['Kub'])*KCAL_PER_KJ/A_PER_NM**2:.4f} {float(p['s0'])*A_PER_NM:.4f} # {name[0]}-{name[1]}-{name[2]}\n")

        f.write("\nDihedral Coeffs\n\n")
        for name, type_id in dihedral_type_map.items():
            terms = filtered_dihedral_coeffs[name]
            num_terms = len(terms)
            line_parts = [f"{type_id}", f"{num_terms}"]
            for p in terms:
                kphi = float(p['Kphi']) * KCAL_PER_KJ
                n = int(float(p['mult']))
                phi0_val = float(p['phi0'])
                line_parts.extend([f"{kphi:.4f}", f"{n}", f"{phi0_val:.0f}"])
            f.write(" ".join(line_parts) + f" # {name[0]}-{name[1]}-{name[2]}-{name[3]}\n")

        # --- Topology (using filtered maps to get correct IDs) ---
        f.write("\nAtoms\n\n")
        for i, atom_coords in enumerate(gro_atoms_mda.positions):
            atom_info = global_atom_map[i + 1]
            f.write(f"{atom_info['id']} {atom_info['mol_id']} {atom_type_map[atom_info['type']]} {atom_info['charge']:.4f} {atom_coords[0]:.4f} {atom_coords[1]:.4f} {atom_coords[2]:.4f}\n")

        f.write("\nBonds\n\n")
        for i, b in enumerate(system_topology['bonds']):
            t1 = global_atom_map[b[0]]['type']
            t2 = global_atom_map[b[1]]['type']
            type_key = tuple(sorted((t1, t2)))
            f.write(f"{i + 1} {bond_type_map[type_key]} {b[0]} {b[1]}\n")

        f.write("\nAngles\n\n")
        for i, a in enumerate(system_topology['angles']):
            t1 = global_atom_map[a[0]]['type']
            t2 = global_atom_map[a[1]]['type']
            t3 = global_atom_map[a[2]]['type']
            type_key = find_angle_type_key(t1, t2, t3, ff_coeffs['angletypes'])
            f.write(f"{i + 1} {angle_type_map[type_key]} {a[0]} {a[1]} {a[2]}\n")
            
        f.write("\nDihedrals\n\n")
        for i, d in enumerate(system_topology['dihedrals']):
            t1 = global_atom_map[d[0]]['type']
            t2 = global_atom_map[d[1]]['type']
            t3 = global_atom_map[d[2]]['type']
            t4 = global_atom_map[d[3]]['type']
            type_key = find_dihedral_type_key(t1, t2, t3, t4, ff_coeffs['dihedraltypes'])
            f.write(f"{i + 1} {dihedral_type_map[type_key]} {d[0]} {d[1]} {d[2]} {d[3]}\n")

def write_lammps_in(in_filename="lammps.in", data_filename="lammps.data"):
    """Writes a basic LAMMPS input script."""
    print(f"--- Writing LAMMPS input file: {in_filename}")
    with open('lammps_template.in', 'r') as fin:
        text = fin.read()
    text = text.replace('{data_filename}', Path(data_filename).name)
    text = text.replace('{trajectory_basename}', Path(data_filename).stem)
    with open(in_filename, 'w') as f:
        f.write(text)

# --- Main Execution ---

def main(gro_file: str, top_file: str, out_folder: Optional[str] = None):
    """Main function to run the conversion."""
    
    parsed_struct = parse_top_level_topology(top_file)

    ff_coeffs = {'atomtypes': {}, 'bondtypes': {}, 'angletypes': {}, 'dihedraltypes': {}}
    for p in parsed_struct['globals']['atomtypes']: ff_coeffs['atomtypes'][p[0]] = {'mass': p[2], 'sigma': p[5], 'epsilon': p[6]}
    for p in parsed_struct['globals']['bondtypes']: ff_coeffs['bondtypes'][tuple(sorted((p[0], p[1])))] = {'b0': p[3], 'Kb': p[4]}
    for p in parsed_struct['globals']['angletypes']: ff_coeffs['angletypes'][(p[0], p[1], p[2])] = {'th0': p[4], 'Kth': p[5], 's0': p[6], 'Kub': p[7]}
    for p in parsed_struct['globals']['dihedraltypes']:
        if len(p) == 7: p.append('1') # Default multiplicity if not specified
        key = (p[0], p[1], p[2], p[3])
        if key not in ff_coeffs['dihedraltypes']: ff_coeffs['dihedraltypes'][key] = []
        ff_coeffs['dihedraltypes'][key].append({'phi0': p[5], 'Kphi': p[6], 'mult': p[7]})

    system_topology = defaultdict(list)
    atom_offset = 0
    mol_instance_counter = 0
    for mol_info in parsed_struct['system']['molecules']:
        mol_name, num_mols = mol_info[0], int(mol_info[1])
        print(f"--- Building {num_mols} molecules of type '{mol_name}'")
        mol_def = parsed_struct['molecules'][mol_name]
        local_atoms_map = {int(p[0]): {'type': p[1], 'charge': float(p[6])} for p in mol_def['atoms']}
        
        for _ in range(num_mols):
            mol_instance_counter += 1
            for local_id, atom_info in local_atoms_map.items():
                system_topology['atoms'].append({'id': local_id + atom_offset, 'mol_id': mol_instance_counter, 'type': atom_info['type'], 'charge': atom_info['charge']})
            for b in mol_def['bonds']: system_topology['bonds'].append([int(b[0]) + atom_offset, int(b[1]) + atom_offset])
            for a in mol_def['angles']: system_topology['angles'].append([int(a[0]) + atom_offset, int(a[1]) + atom_offset, int(a[2]) + atom_offset])
            for d in mol_def['dihedrals']: system_topology['dihedrals'].append([int(d[0]) + atom_offset, int(d[1]) + atom_offset, int(d[2]) + atom_offset, int(d[3]) + atom_offset])
            atom_offset += len(local_atoms_map)

    gro_atoms_mda, box_dims = parse_gro_mda(gro_file)
    if len(system_topology['atoms']) != len(gro_atoms_mda):
        raise ValueError(f"Atom count mismatch: Topology has {len(system_topology['atoms'])}, GRO file has {len(gro_atoms_mda)}.")

    if out_folder is None: out_folder = dirname(gro_file)
    os.makedirs(out_folder, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(gro_file))[0]
    lammps_data_filename = os.path.join(out_folder, f"{base_name}.data")
    lammps_in_filename = os.path.join(out_folder, f"{base_name}.in")
    
    write_lammps_data(ff_coeffs, system_topology, gro_atoms_mda, box_dims, data_filename=lammps_data_filename)
    write_lammps_in(in_filename=lammps_in_filename, data_filename=lammps_data_filename)

    print(f"\nConversion complete. Generated '{lammps_data_filename}' and '{lammps_in_filename}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert LAMMPS XYZ trajectory with integer types to element symbols based on mass."
    )
    parser.add_argument(
        "gro", 
        help="Path to the GROMACS .gro file."
    )
    parser.add_argument(
        "top", 
        help="Path to the GROMACS .top file."
    )
    parser.add_argument(
        "-o", "--output", 
        default=None,
        help="Path for the output folder, where .data and .in converted files will be saved. Defaults to same folder of gro file."
    )
    
    args = parser.parse_args()

    main(gro_file=args.gro, top_file=args.top, out_folder=args.output)