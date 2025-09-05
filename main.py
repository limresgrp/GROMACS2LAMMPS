"""
Main executable script for the GROMACS to LAMMPS converter.

This script orchestrates the conversion process by:
1. Parsing command-line arguments for input/output files.
2. Calling the GROMACS parser to read topology and structure files.
3. Calling the system builder to construct the full system from molecule definitions.
4. Calling the LAMMPS writer to generate the final .data and .in files.
"""

import argparse
import os
from pathlib import Path
from typing import Optional
import MDAnalysis as mda

# Import modules from the converter package
from gmx_parser import GromacsParser
from system_builder import SystemBuilder
from lammps_writer import LammpsWriter

def main(gro_file: str, top_file: str, out_folder: Optional[str] = None):
    """Main function to run the conversion."""
    print("=====================================================")
    print("=== GROMACS to LAMMPS Conversion Script           ===")
    print("=====================================================\n")

    # --- 1. Parsing ---
    print("\n--- [Step 1/4] Parsing GROMACS topology and structure files ---")
    parser = GromacsParser(top_file)
    parsed_ff = parser.parse()
    
    gro_universe = mda.Universe(gro_file)
    gro_atoms = gro_universe.atoms
    box_dims = gro_universe.dimensions[:3]
    print(f"Successfully parsed '{top_file}' and '{gro_file}'.")
    
    # --- 2. System Building ---
    print("\n--- [Step 2/4] Building full system topology from molecules ---")
    builder = SystemBuilder(parsed_ff)
    system_topology = builder.build()
    
    if len(system_topology['atoms']) != len(gro_atoms):
        raise ValueError(
            f"Atom count mismatch: Topology has {len(system_topology['atoms'])}, "
            f"GRO file has {len(gro_atoms)}."
        )
    print(f"System built successfully with {len(system_topology['atoms'])} atoms.")

    # --- 3. Writing Output ---
    print("\n--- [Step 3/4] Preparing and writing LAMMPS output files ---")
    if out_folder is None: 
        out_folder = Path(gro_file).parent
    os.makedirs(out_folder, exist_ok=True)
    
    base_name = Path(gro_file).stem
    lammps_data_filename = Path(out_folder) / f"{base_name}.data"
    lammps_in_filename = Path(out_folder) / f"{base_name}.in"
    
    writer = LammpsWriter(parsed_ff, system_topology, gro_atoms, box_dims)
    writer.write_lammps_data(lammps_data_filename)
    writer.write_lammps_in(lammps_in_filename)

    # --- 4. Completion ---
    print("\n--- [Step 4/4] Conversion complete! ---")
    print(f"Generated LAMMPS files:")
    print(f"  - Data file: {lammps_data_filename}")
    print(f"  - Input file: {lammps_in_filename}")
    print("\n=====================================================")


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser(
        description="Convert GROMACS .gro and .top files to LAMMPS data and input files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    cli_parser.add_argument(
        "gro", 
        help="Path to the GROMACS coordinate file (.gro)."
    )
    cli_parser.add_argument(
        "top", 
        help="Path to the top-level GROMACS topology file (.top)."
    )
    cli_parser.add_argument(
        "-o", "--output", 
        default=None,
        help="Path for the output folder. Defaults to the same folder as the .gro file."
    )
    
    args = cli_parser.parse_args()
    main(gro_file=args.gro, top_file=args.top, out_folder=args.output)
