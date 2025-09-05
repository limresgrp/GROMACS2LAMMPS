"""
System Builder Module.

This module takes the parsed GROMACS data structure and constructs a
flat list of all atoms, bonds, angles, and dihedrals for the entire system.

Key functionalities:
- Iterates through the [ molecules ] section of the topology.
- Expands each molecule type into the specified number of instances.
- Correctly offsets atom IDs for each new molecule instance.
- Creates a unified system topology ready for writing to a LAMMPS data file.
"""
from collections import defaultdict
from typing import Dict, Any

class SystemBuilder:
    """Builds the complete system topology from parsed molecule definitions."""

    def __init__(self, parsed_data: Dict[str, Any]):
        """
        Initializes the builder.

        Args:
            parsed_data: The dictionary structure produced by GromacsParser.
        """
        self.parsed_data = parsed_data
        self.system_topology = defaultdict(list)

    def build(self) -> Dict[str, Any]:
        """
        Constructs the full system topology.

        Returns:
            A dictionary containing flat lists of 'atoms', 'bonds', 'angles', etc.
        """
        atom_offset = 0
        mol_instance_counter = 0
        
        if not self.parsed_data['system']['molecules']:
            raise ValueError("No molecules found in the [ molecules ] section of the topology.")

        for mol_info in self.parsed_data['system']['molecules']:
            mol_name, num_mols = mol_info[0], int(mol_info[1])
            print(f"  -> Building {num_mols} instances of molecule '{mol_name}'")
            
            mol_def = self.parsed_data['molecules'].get(mol_name)
            if not mol_def:
                raise KeyError(f"Molecule type '{mol_name}' is defined in [system] but not found.")

            # Create a local map of atom IDs to their info within one molecule
            local_atoms_map = {
                int(p[0]): {'type': p[1], 'charge': float(p[6])} 
                for p in mol_def.get('atoms', [])
            }
            if not local_atoms_map:
                raise ValueError(f"Molecule '{mol_name}' has no [atoms] defined.")

            for _ in range(num_mols):
                mol_instance_counter += 1
                
                # Add atoms, offsetting IDs
                for local_id, atom_info in local_atoms_map.items():
                    self.system_topology['atoms'].append({
                        'id': local_id + atom_offset, 
                        'mol_id': mol_instance_counter, 
                        'type': atom_info['type'], 
                        'charge': atom_info['charge']
                    })

                # Add bonds, angles, dihedrals with offset IDs
                self._add_topology('bonds', mol_def, atom_offset, 2)
                self._add_topology('angles', mol_def, atom_offset, 3)
                self._add_topology('dihedrals', mol_def, atom_offset, 4)

                atom_offset += len(local_atoms_map)
                
        return self.system_topology

    def _add_topology(self, section: str, mol_def: Dict, offset: int, num_atoms: int):
        """Helper to add bonded terms with the correct atom ID offset."""
        for items in mol_def.get(section, []):
            atom_indices = [int(items[i]) + offset for i in range(num_atoms)]
            self.system_topology[section].append(atom_indices)
