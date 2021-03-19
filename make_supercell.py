import sys
import numpy
from math import sin, cos, sqrt, pi, acos 
from numpy import array, add, dot, cross, arccos

class SuperCell:
    def __init__(self, initial_file, xA, xB, xC, output_file):
        self._initial_file = initial_file
        self._xA = xA
        self._xB = xB
        self._xC = xC
        self._output_file = output_file

    def perpvector(self, v1, v2, v3):
        J = cross(v2 - v3, v2 - v1)+ v2
        J = (J - v2) /(sqrt(dot(J - v2, J - v2))) + v2
        return J
    
    def rotate(self, V, J, T):
        x = V[0]
        y = V[1]
        z = V[2]
        u = J[0]
        v = J[1]
        w = J[2]
        a = (u*(u*x + v*y + w*z) + (x * (v*v + w*w) - u *(v*y + w*z))*cos(T) + sqrt(u*u + v*v + w*w)*(-w*y + v*z)*sin(T))/(u*u + v*v + w*w)
        b = (v*(u*x + v*y + w*z) + (y * (u*u + w*w) - v *(u*x + w*z))*cos(T) + sqrt(u*u + v*v + w*w)*(w*x - u*z)*sin(T))/(u*u + v*v + w*w)
        c = (w*(u*x + v*y + w*z) + (z * (u*u + v*v) - w *(u*x + v*y))*cos(T) + sqrt(u*u + v*v + w*w)*(-v*x + u*y)*sin(T))/(u*u + v*v + w*w)
        return array([a, b, c])

    def vector_angle(self, a, b, c):
    
        # In case numpy.dot() returns larger than 1
        # and we cannot take acos() to that number
        acos_out_of_bound = 1.0
        v1 = a - b
        v2 = c - b
        v1 = v1 / sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        v2 = v2 / sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
        dot_product = dot(v1,v2)
    
        if dot_product > acos_out_of_bound:
            dot_product = acos_out_of_bound
        if dot_product < -1.0 * acos_out_of_bound:
            dot_product = -1.0 * acos_out_of_bound
    
        return arccos(dot_product)
        
    def norm_vector(self, vector):
    
        length = sqrt((vector[0]*vector[0]) + (vector[1]*vector[1]) + (vector[2]*vector[2]))
        norm_vector = array([vector[0]/length,vector[1]/length,vector[2]/length])

        return norm_vector
    
    def get_unit_cell_vectors_GMX(self, collinear_vector_length, plane_vector_length, free_vector_length, alpha, beta, gamma):
    
        alpha = numpy.deg2rad(alpha)
        beta  = numpy.deg2rad(beta)
        gamma = numpy.deg2rad(gamma)
    
        collinear_vector = numpy.array([collinear_vector_length,0,0])
    
        plane_vector = numpy.array([plane_vector_length*cos(gamma), plane_vector_length*sin(gamma), 0])
    
        free_vector     = numpy.array([free_vector_length*cos(beta),0 ,0])
        free_vector[1]  = (free_vector_length * (cos(alpha) - cos(beta)*cos(gamma)))/(sin(gamma))
        free_vector[2]  = sqrt(free_vector_length*free_vector_length - free_vector[0]*free_vector[0] - free_vector[1]*free_vector[1])
    
        return collinear_vector, plane_vector, free_vector

    def fractional_to_cartesian(self, vector, lattice_a, lattice_b, lattice_c):

        # converts numpy vector from direct lattice to cartesian

        # get vector length
        a_length = numpy.linalg.norm(lattice_a)
        b_length = numpy.linalg.norm(lattice_b)
        c_length = numpy.linalg.norm(lattice_c)

        # calculate lattice vector angles
        alpha = acos(dot(lattice_b,lattice_c)/(b_length*c_length))
        beta = acos(dot(lattice_a,lattice_c)/(a_length*c_length))
        gamma = acos(dot(lattice_b,lattice_a)/(b_length*a_length))

        # calculate volume
        v = sqrt(1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + 2*cos(alpha)*cos(beta)*cos(gamma) )

        conversion_matrix =     array([
                                      [ a_length, b_length*cos(gamma), c_length*cos(beta)],
                                      [ 0       , b_length*sin(gamma), c_length*((cos(alpha)-cos(beta)*cos(gamma))/(sin(gamma)))],
                                      [ 0       ,                   0, c_length*((v)/(sin(gamma)))],
                                      ])
        # calculate cartesian vector
        cartesian_vector = dot(conversion_matrix,vector)

        return cartesian_vector

    def cartesian_to_fractional(self, vector, lattice_a, lattice_b, lattice_c):

        # converts numpy vector from cartesian to fractional lattice

        # get vector length
        a_length = np.linalg.norm(lattice_a)
        b_length = np.linalg.norm(lattice_b)
        c_length = np.linalg.norm(lattice_c)

        # calculate lattice vector angles
        alpha = math.acos(np.dot(lattice_b,lattice_c)/(b_length*c_length))
        beta = math.acos(np.dot(lattice_a,lattice_c)/(a_length*c_length))
        gamma = math.acos(np.dot(lattice_b,lattice_a)/(b_length*a_length))

        # calculate volume
        v = math.sqrt(1 - math.cos(alpha)**2 - math.cos(beta)**2 - math.cos(gamma)**2 + 2*math.cos(alpha)*math.cos(beta)*math.cos(gamma) )

        # make conversion matrix
        conversion_matrix = np.array([
                             [  1/a_length , -( math.cos(gamma) )/( a_length*math.sin(gamma)) , ( math.cos(alpha)*math.cos(gamma)-math.cos(beta) )/( a_length*v*math.sin(gamma))],
                             [           0 ,                     1/(b_length*math.sin(gamma)) , ( math.cos(beta)*math.cos(gamma)-math.cos(alpha) )/( b_length*v*math.sin(gamma))],
                             [           0 ,                                                0 ,                                                     math.sin(gamma)/(c_length*v)],
                             ])

        # calculate fractional vector
        direct_vector = np.dot(conversion_matrix,vector)

        return direct_vector



    def read_cif(self, cif_file):

        f = open(cif_file, 'r')
        alllines = []
        for line in f:
            alllines.append(line)
        f.close()

        A = ''
        B = ''
        C = ''
        alpha = ''
        beta = ''
        gamma = ''
        information = []
        atoms = []
        unitcell_loop = []
        atom_loop = []

        LOOPS = []

        for line in alllines:
            if 'loop_' in line:
                break
            information.append(line)

        for i in range(0, len(alllines)):
            if 'loop_' in alllines[i]:
                loop = ['loop_']
                for line in alllines[i+1:len(alllines)]:
                    if 'loop_' in line:
                        LOOPS.append(loop)
                        loop = []
                        break

                    elif line == alllines[-1]:
                        loop.append(line)
                        LOOPS.append(loop)
                        loop = []
                        break
                    else:
                        loop.append(line)
        cell_loop = []
        atom_loop = []
        for loop in LOOPS:
            for line in loop:
                if 'cell_length' in line:
                    cell_loop = loop
                    break
                if 'atom_site_fract' in line:
                    atom_loop = loop
                    break

        for line in cell_loop:
            if '_cell_length_a' in line:
                A = float(line.replace('\n','').split()[1].split('(')[0])
            if '_cell_length_b' in line:
                B = float(line.replace('\n','').split()[1].split('(')[0])
            if '_cell_length_c' in line:
                C = float(line.replace('\n','').split()[1].split('(')[0])
            if '_cell_angle_alpha' in line:
                alpha = float(line.replace('\n','').split()[1].split('(')[0])
            if '_cell_angle_beta' in line:
                beta = float(line.replace('\n','').split()[1].split('(')[0])
            if '_cell_angle_gamma' in line:
                gamma = float(line.replace('\n','').split()[1].split('(')[0])
        unitcell = [A, B, C, alpha, beta, gamma]

        for line in atom_loop:
            if len(line.split()) > 1:
                # atomtype atomname molname x y z
                line = line.split()

                atom = [line[0], line[1], 'MOL', float(line[2].split('(')[0]), float(line[3].split('(')[0]), float(line[4].split('(')[0])]
                #atom = [line[0], line[1], 'MOL', float(line[2]), float(line[3]), float(line[4])]
                atoms.append(atom)

        return atoms, unitcell, information

    def read_pdb(self, pdb_file):
        # to do
        pass


    def generate_symmetry(self):
        # to do
        test =  ['1 x,y,z']
        test += ['2 1/2-x,y,1/2-z']
        test += ['3 -x,-y,-z']
        test += ['4 1/2+x,-y,1/2+z']
        test += ['5 x,1/2+y,1/2+z']
        test += ['6 1/2-x,1/2+y,-z']
        test += ['7 -x,1/2-y,1/2-z']
        test += ['8 1/2+x,1/2-y,z']


    def make_supercell_fractional(self, atoms, unitcell, xA, xB, xC):
        # (0.5 / 10)+9*(1/10)
        supercell_atoms = []
        for a in range(0,xA):
            for b in range(0,xB):
                for c in range(0,xC):
                    for atom in atoms:
                        atom_A = (atom[3] / float(xA)) + a*(1/float(xA))
                        atom_B = (atom[4] / float(xB)) + b*(1/float(xB))
                        atom_C = (atom[5] / float(xC)) + c*(1/float(xC))
                        newatom = [atom[0], atom[1], atom[2], atom_A, atom_B, atom_C]
                        supercell_atoms.append(newatom)

        unitcell[0] = unitcell[0]*xA
        unitcell[1] = unitcell[1]*xB
        unitcell[2] = unitcell[2]*xC

        return supercell_atoms, unitcell

    def fractional_atoms_to_cartesian(self, atoms, unitcell):
        newatoms = []

        for atom in atoms:
            ATOM_fractional = array([atom[3], atom[4], atom[5]])
            A, B, C = self.get_unit_cell_vectors_GMX(unitcell[0], unitcell[1], unitcell[2], unitcell[3], unitcell[4], unitcell[5])
            XYZ = self.fractional_to_cartesian(ATOM_fractional, A, B, C) 
            newatom = [atom[0], atom[1], atom[2], XYZ[0], XYZ[1], XYZ[2]]
            newatoms.append(newatom)

        return newatoms

    def write_cif(atoms, A, B, C, filename):
        pass 

    def get_pdbatomstring(self, ATOM_NR, ATOM_TYPE, RES_NAME, CHAIN, MOL_ID, ACHAR, X, Y, Z, OCCU, TEMP_FACTOR, ID, ELEMENT, CHARGE):
        STRING = "ATOM  " + "%5s %4s%1s%3s%2s%4s%4s%8.3f%8.3f%8.3f%6.2f%6.2f      %4s%2s%2s" % (ATOM_NR, ATOM_TYPE, ' ', RES_NAME, '  ', MOL_ID, ACHAR, X, Y, Z, OCCU, TEMP_FACTOR, ID, ELEMENT, CHARGE  )
        #print [STRING]
        return STRING

    def write_pdb(self, atoms, unitcell, filename):

        f = open(filename, 'w')
        out  = 'REMARK pdb created with make_supercell.py\n'
        out += 'CRYST1 '+str(unitcell[0])+' '+str(unitcell[1])+' '+str(unitcell[2])+' '+str(unitcell[3])+' '+str(unitcell[4])+' '+str(unitcell[5])+'\n' 
        
        f.write(out)        

        atom_nr = 1
        for atom in atoms:
            LINE = self.get_pdbatomstring( atom_nr, atom[0], atom[2], '', '1', '', atom[3], atom[4], atom[5], 0.00, 0.00, '', atom[1], '')+'\n'
            atom_nr += 1

            if atom_nr == 100000:
                print ('Warning atom numbers over 100K. Atom number will start over from 0')
                atom_nr = 0

            f.write(LINE)
        LINE = 'TER '+str(atom_nr)+'\n'
        f.write(LINE)
        f.close()

    def write_gro(self, atoms, unitcell, filename):

        f = open(filename, 'w')
        f.write('Made with make_supercell.py\n')

        number_of_atoms = len(atoms)
        f.write(str(number_of_atoms)+'\n')

        # %5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f
   
        res_id = 1
        atom_nr = 1
        for atom in atoms:
            line = "%5i%-5s%5s%5i%8.3f%8.3f%8.3f\n" % (res_id, atom[2], atom[0], atom_nr, atom[3], atom[4], atom[5] ) 
            f.write(line)            

            atom_nr += 1
            if atom_nr == 100000:
                print ('Warning atom numbers over 100K. Atom number will start over from 0')
                atom_nr = 0


        collinear_vector, plane_vector, free_vector = self.get_unit_cell_vectors_GMX(unitcell[0], unitcell[1], unitcell[2], unitcell[3], unitcell[4], unitcell[5])

        unitcell_line = "%8.5f  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f\n" % (collinear_vector[0]/10.0, plane_vector[1]/10.0, free_vector[2]/10.0, collinear_vector[1]/10.0, collinear_vector[2]/10.0, plane_vector[0]/10.0, plane_vector[2]/10.0, free_vector[0]/10.0, free_vector[1]/10.0)
        f.write(unitcell_line)
        f.close()
 
    def aangstrom_to_nanometers(self, atoms):
        new_atoms = []
        for atom in atoms:
            new_atom = atom
            new_atom[3] = new_atom[3] / 10.0
            new_atom[4] = new_atom[4] / 10.0
            new_atom[5] = new_atom[5] / 10.0
            new_atoms.append(new_atom)
        return new_atoms

    def run(self):
        atoms = []
        A = None
        B = None
        C = None
        if '.cif' in self._initial_file:
            atoms, unitcell, information = self.read_cif(self._initial_file)
         #   self.generate_symmetry()
            supercell_atoms, newunitcell = self.make_supercell_fractional(atoms, unitcell, self._xA, self._xB, self._xC)
            if '.cif' in self._output_file:
                seld.write_cif(supercell_atoms, newunitcell, self._output_file)
            elif '.pdb' in self._output_file:
                cartesian_supercellatoms = self.fractional_atoms_to_cartesian(supercell_atoms, newunitcell)        
                self.write_pdb(cartesian_supercellatoms, newunitcell, self._output_file)
            elif '.gro' in self._output_file:
                cartesian_supercellatoms = self.fractional_atoms_to_cartesian(supercell_atoms, newunitcell)        
                nanometer_atoms = self.aangstrom_to_nanometers(cartesian_supercellatoms)
                self.write_gro(nanometer_atoms, newunitcell, self._output_file)
            else:
                print ('Unrecognised format in ' + self._output_file)

        elif '.pdb' in self._intial_file:
            atoms, unitcell = self.read_pdb(self._intial_file)
        else:
            print ('ERROR unrecognised file format! must be .cif or .pdb')
            sys.exit(0)
        

# use as input.cif 2 2 2 output.pdb
initial_file = sys.argv[1]
xA = int(sys.argv[2])
xB = int(sys.argv[3])
xC = int(sys.argv[4])
output_file = 'temp.cif'
if len(sys.argv) > 5:
    output_file = sys.argv[5]

supercell = SuperCell(initial_file, xA, xB, xC, output_file)
supercell.run()
