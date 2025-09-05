"""
GROMACS Topology Parser Module.

This module is responsible for reading and parsing a GROMACS topology,
including all #included .itp files. It organizes force field parameters
and molecule definitions into a structured dictionary.

Key functionalities:
- Recursively follows #include directives.
- Scopes parameters within [ moleculetype ] blocks.
- Parses [ defaults ], [ atomtypes ], bonded, and non-bonded parameters.
- Handles basic #ifdef/#endif directives for conditional parsing.
"""

import os
import re
from collections import defaultdict
from typing import Set, Dict, Any

class GromacsParser:
    """Parses a GROMACS topology hierarchy into a structured dictionary."""

    def __init__(self, top_level_file: str):
        """
        Initializes the parser.
        
        Args:
            top_level_file: Path to the main .top file.
        """
        if not os.path.exists(top_level_file):
            raise FileNotFoundError(f"Top-level topology file not found: {top_level_file}")
        self.top_level_file = top_level_file
        self.data_struct: Dict[str, Any] = {
            "globals": defaultdict(list),
            "molecules": {},
            "system": defaultdict(list)
        }
        self.visited_files: Set[str] = set()

    def parse(self) -> Dict[str, Any]:
        """Entry point to start the parsing process."""
        print(f"Starting topology parsing from: {self.top_level_file}")
        self._recursive_parse(self.top_level_file)
        self._process_dihedral_types()
        return self.data_struct

    def _recursive_parse(self, filepath: str):
        """
        Recursively parses a GROMACS topology file (.top or .itp).
        
        Args:
            filepath: The path to the file to parse.
        """
        abs_path = os.path.abspath(filepath)
        if abs_path in self.visited_files:
            return
        
        print(f"  -> Reading file: {os.path.basename(filepath)}")
        self.visited_files.add(abs_path)
        
        current_file_dir = os.path.dirname(abs_path)
        current_section: str | None = None
        current_mol_type: str | None = None
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
                    self._recursive_parse(include_path)
                    continue

                match = re.match(r'\[\s*(\w+)\s*\]', line)
                if match:
                    current_section = match.group(1).lower()
                    if current_section == 'moleculetype':
                        current_mol_type = None
                    continue

                if not current_section:
                    continue

                if current_section == 'moleculetype':
                    if not current_mol_type:
                        current_mol_type = line.split()[0]
                        if current_mol_type not in self.data_struct['molecules']:
                            self.data_struct['molecules'][current_mol_type] = defaultdict(list)
                elif current_section in ['system', 'molecules']:
                    self.data_struct['system'][current_section].append(line.split())
                elif current_mol_type:
                    # Parameter is scoped inside a molecule definition
                    self.data_struct['molecules'][current_mol_type][current_section].append(line.split())
                else:
                    # Parameter is global (e.g., in a forcefield.itp file)
                    self.data_struct['globals'][current_section].append(line.split())

    def _process_dihedral_types(self):
        """
        Processes the raw dihedral types to group them by atom types.
        This handles multi-term Fourier dihedrals.
        """
        dihedrals_by_key = defaultdict(list)
        raw_dihedrals = self.data_struct['globals'].get('dihedraltypes', [])
        
        for p in raw_dihedrals:
            if len(p) < 6: continue
            key = (p[0], p[1], p[2], p[3])
            func_type = int(p[4])
            params = p[5:]
            
            # GROMACS multi-term Fourier (funct 9) and Ryckaert-Bellemans (funct 3)
            # are common and map to different LAMMPS styles.
            if func_type in [9, 3, 4, 1]:
                 dihedrals_by_key[key].append({'func': func_type, 'params': params})

        self.data_struct['globals']['processed_dihedrals'] = dihedrals_by_key
