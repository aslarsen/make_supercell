# make_supercell
Reads a .cif file and creates a new supercell in the formats .pdb or .gro. Use as python input.cif 2 2 2 output.pdb
The .cif must be saved with all atoms in the unit cell and not just the asymmetric unit. If an atom is placed directly on the unit cell then it risk being duplicated and causing atom overlap.
