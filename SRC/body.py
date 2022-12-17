from math import sqrt
import numpy as np
import astropy.units as astr_un
from astropy import constants as astr_const


class Body():
    '''
    Body class for particle, used in Simulaion
    ------------------------------------------
    Params:
        mass: mass of particle.
        x_vec: vector len(3) containing the x, y, z positions.
        v_vec: vector len(3) containing the v_x, v_y, v_z velocities.
        name: string, name of body.
        has_units: type of dimension?

    Example of using:    
        Mars = Body(name='Mars', x_vec=mars_x, v_vec=mars_v, mass=mars_mass)
    '''
    def __init__(self, mass, x_vec, v_vec, name='Unknown', has_units=None):
        self.name = name
        self.has_units = has_units
        if self.has_units and self.has_units.upper() == 'CGS':
            self.mass = mass.cgs
            self.x_vec = x_vec.cgs.value
            self.v_vec = v_vec.cgs.value
        elif self.has_units and self.has_units.upper() == 'SI':
            self.mass = mass.si
            self.x_vec = x_vec.si.value
            self.v_vec = v_vec.si.value          
        else:
            self.mass = mass
            self.x_vec = x_vec
            self.v_vec = v_vec

    def return_vec(self):
        '''
        return concatenates x and v into "y" vector of positions and velocities
        '''
        return np.concatenate((self.x_vec, self.v_vec))
    
    def return_mass(self):
        if self.has_units and self.has_units.upper() == 'CGS':
            return self.mass.cgs.value
        elif self.has_units and self.has_units.upper() == 'SI':
            return self.mass.si.value
        else:
            return self.mass
    
    def return_name(self):
        return self.name