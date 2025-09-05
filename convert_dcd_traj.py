import argparse
import os
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.transformations.base import TransformationBase

def create_type_map_from_mass(data_file: str) -> dict:
    """
    Creates a mapping from LAMMPS integer atom type to an element symbol
    by finding the closest element based on mass from the data file.
    """
    print("--- Creating atom type map from masses ---")
    
    element_masses = {
        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 
        'F': 18.998, 'P': 30.974, 'S': 32.06, 'Cl': 35.45
    }
    
    lammps_type_to_mass = {}
    in_masses_section = False
    
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if 'Masses' in line:
                in_masses_section = True
                continue
            if in_masses_section and not line.split('#')[0].strip():
                continue
            if in_masses_section and line and not line[0].isdigit():
                break
            
            if in_masses_section:
                parts = line.split()
                if len(parts) >= 2:
                    lammps_type = int(parts[0])
                    mass = float(parts[1])
                    lammps_type_to_mass[lammps_type] = mass

    if not lammps_type_to_mass:
        raise ValueError(f"Could not find a valid 'Masses' section in {data_file}")

    final_map = {}
    for lammps_type, mass in lammps_type_to_mass.items():
        closest_element = min(element_masses, key=lambda k: abs(element_masses[k] - mass))
        final_map[lammps_type] = closest_element
        print(f"  Mapping LAMMPS type {lammps_type} (mass {mass:.3f}) -> Element '{closest_element}'")
        
    return final_map


def process_trajectory(
    data_file: str, 
    dcd_file: str, 
    output_traj_file: str, 
    output_structure_file: str = None,
    stride: int = 1, 
    align_selection: str = None,
    center_x_selection: str = None,
    center_y_selection: str = None,
    center_z_selection: str = None
):
    """
    Loads a LAMMPS data/dcd trajectory, processes it, and writes to new file(s).
    """
    print("--- Loading trajectory ---")
    try:
        # Corrected format from 'LAMMPS' to 'LAMMPSDCD'
        universe = mda.Universe(data_file, dcd_file, format='LAMMPS')
    except Exception as e:
        print(f"Error loading files: {e}")
        print("Please ensure your LAMMPS data file contains 'Masses' and 'Atoms' sections.")
        return

    type_map = create_type_map_from_mass(data_file)
    universe.add_TopologyAttr('elements', [type_map.get(t, 'X') for t in universe.atoms.types.astype(int)])

    print(f"Loaded {universe.atoms.n_atoms} atoms and {universe.trajectory.n_frames} frames.")

    # --- Setup Transformations ---
    transformations = []
    from MDAnalysis.transformations import unwrap, wrap

    # --- Centering Logic ---
    center_dims = []
    center_selection_str = None
    
    # Consolidate centering requests
    if center_x_selection:
        center_dims.append('x')
        center_selection_str = center_x_selection
    if center_y_selection:
        center_dims.append('y')
        if center_selection_str and center_selection_str != center_y_selection:
            raise ValueError("Mismatched selection strings for different centering axes.")
        center_selection_str = center_y_selection
    if center_z_selection:
        center_dims.append('z')
        if center_selection_str and center_selection_str != center_z_selection:
            raise ValueError("Mismatched selection strings for different centering axes.")
        center_selection_str = center_z_selection

    if center_selection_str:
        print(f"Enabled transformation: Centering on selection '{center_selection_str}' for axes {center_dims}.")

        # Define a custom transformation for per-axis centering
        class CenterSelection(TransformationBase):
            def __init__(self, atomgroup, center_dims, **kwargs):
                super().__init__(**kwargs)
                self.ag = atomgroup
                self.center_dims = center_dims
                
            def _transform(self, ts):
                # CM of the selection (which should be whole from a previous unwrap)
                selection_cm = self.ag.center_of_mass()
                box_center = ts.dimensions[:3] / 2.0
                translation_vector = box_center - selection_cm
                
                if 'x' not in self.center_dims: translation_vector[0] = 0
                if 'y' not in self.center_dims: translation_vector[1] = 0
                if 'z' not in self.center_dims: translation_vector[2] = 0
                
                self.ag.universe.atoms.positions += translation_vector
                return ts

        # Build the robust centering workflow as requested
        center_group = universe.select_atoms(center_selection_str)
        transformations.extend([
            unwrap(universe.atoms),
            CenterSelection(center_group, center_dims),
            wrap(universe.atoms),
            unwrap(universe.atoms)
        ])
    else:
        # If no centering, still make molecules whole
        transformations.append(unwrap(universe.atoms))

    # --- Alignment Logic (applied after centering) ---
    if align_selection:
        print(f"Enabled transformation: Aligning trajectory on selection '{align_selection}'")
        align.AlignTraj(universe, universe, select=align_selection, in_memory=True,
                        transform_workflow=transformations).run()
        # After running the alignment, the transformations are already applied.
        # We clear the list to avoid applying them a second time.
        transformations.clear()

    # --- Writing Output ---
    all_atoms = universe.select_atoms("all")
    workflow = [tr for tr in transformations if tr is not None]
    
    if output_structure_file:
        print(f"--- Writing structure file to {output_structure_file} ---")
        # Go to the first frame to write the structure
        universe.trajectory[0]
        # Apply the full workflow to this single frame before writing
        for tr in workflow:
            tr(universe.trajectory.ts)
        all_atoms.write(output_structure_file)

    print(f"--- Writing trajectory to {output_traj_file} ---")
    with mda.Writer(output_traj_file, all_atoms.n_atoms) as writer:
        for ts in universe.trajectory[::stride]:
            # If alignment was run, transformations are already done.
            # Otherwise, apply them now.
            if not align_selection:
                for tr in workflow:
                    tr(ts)
            writer.write(all_atoms)
            
    print(f"Successfully wrote {len(universe.trajectory[::stride])} frames to {output_traj_file}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Post-process a LAMMPS DCD trajectory. This script can make molecules whole, align the trajectory, and stride the output.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("data_file", help="Path to the LAMMPS .data file (used for topology and masses).")
    parser.add_argument("dcd_file", help="Path to the input LAMMPS .dcd trajectory file.")
    parser.add_argument(
        "-o", "--output", default="output.xtc",
        help="""Path for the output file. Format is inferred from the extension.
Recommended formats: .pdb, .xyz, .xtc
SPECIAL BEHAVIOR: If the extension is '.xtc', a companion '.pdb'
file with the same base name will be created automatically for the structure.
Default: output.xtc"""
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame step to stride the trajectory. '1' means every frame, '10' means every 10th frame. Default: 1")
    parser.add_argument(
        "--align", type=str, metavar="'SELECTION'", default=None,
        help="""(Optional) Align the trajectory based on a selection string.
The trajectory is aligned to the first frame.
Examples:
  --align 'protein and name CA'
  --align 'resname POPC and not name H*'"""
    )
    parser.add_argument("--center-x", type=str, metavar="'SELECTION'", default=None, help="(Optional) Center on selection in X.")
    parser.add_argument("--center-y", type=str, metavar="'SELECTION'", default=None, help="(Optional) Center on selection in Y.")
    parser.add_argument("--center-z", type=str, metavar="'SELECTION'", default=None, help="(Optional) Center on selection in Z.")
    
    args = parser.parse_args()

    output_traj_file = args.output
    output_structure_file = None

    if args.output.lower().endswith('.xtc'):
        base_name = os.path.splitext(args.output)[0]
        output_structure_file = base_name + '.pdb'
        print(f"INFO: Output is XTC. A separate structure file will be saved: {output_structure_file}")

    process_trajectory(
        data_file=args.data_file,
        dcd_file=args.dcd_file,
        output_traj_file=output_traj_file,
        output_structure_file=output_structure_file,
        stride=args.stride,
        align_selection=args.align,
        center_x_selection=args.center_x,
        center_y_selection=args.center_y,
        center_z_selection=args.center_z
    )