from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as astr_un
from astropy import constants as astr_const
import sys
import time
from enum import Enum
from numba import njit
from mpl_toolkits.mplot3d import Axes3D
import datetime
import os


@njit
def kahan_sum(input_vec):
    sum_ = 0.0
    c = 0.0
    for i in range(len(input_vec)):
        y = input_vec[i] - c
        t = sum_ + y
        c = (t - sum_) - y
        sum_ = t
    return sum_

@njit
def calc_tol_(y, y_, a_tol, r_tol, **kwargs):
    if np.isscalar(a_tol) and np.isscalar(r_tol):
        return np.array([a_tol+max(abs(y[i]), abs(y_[i]))*r_tol for i in range(len(y))])

    if len(a_tol) == len(r_tol) == 1:
        return np.array([a_tol[0]+max(abs(y[i]), abs(y_[i]))*r_tol[0] for i in range(len(y))])


def calc_tol_n(y, y_, a_tol, r_tol, **kwargs):
    if 'nbody' in kwargs:
        nbody = kwargs['nbody']
    else:
        raise ValueError('You need to set "nbody" in calc_tol_n()!')

    if len(a_tol) == len(r_tol) == 2:
        res = np.zeros(len(y))
        for i in range(nbody):
            offset = i * 6
            res[offset] = a_tol[0] + max(abs(y[offset]), abs(y_[offset]))*r_tol[0]
            res[offset+1] = a_tol[0] + max(abs(y[offset+1]), abs(y_[offset+1]))*r_tol[0]
            res[offset+2] = a_tol[0] + max(abs(y[offset+2]), abs(y_[offset+2]))*r_tol[0]

            res[offset+3] = a_tol[1] + max(abs(y[offset+3]), abs(y_[offset+3]))*r_tol[1]
            res[offset+4] = a_tol[1] + max(abs(y[offset+4]), abs(y_[offset+4]))*r_tol[1]
            res[offset+5] = a_tol[1] + max(abs(y[offset+5]), abs(y_[offset+5]))*r_tol[1]
        return res

    raise AttributeError('Error size of atol/rtol!')


@njit
def calc_err_(tol, y, y_):
    res = np.array([abs(y[i]-y_[i])/tol[i] for i in range(y.size)])
    return res



def calc_err_norm(err, ord=None):
    '''
    Function for calculate one of the types of norms for err vector.
    ------------------------------
    Params:
        err: vector of errors / tol_i.
        ord: order of the norm (type of norm).
            ============    ===============================================================
            None            calculated as in a lecture: sqrt{1/n * sum[(xi - x_i) / tol_i]}
            'std'           similarly None
            2/'2'/'eucl'    euclidean norm
            inf/'oo'        infinite norm
            1/'1'           unit norm: sum[abs(xi)]
    '''
    if ord==None or ord=='std':
        res = np.sum(err**2) / err.size
        return sqrt(res)
    
    if ord==2 or ord=='2' or ord=='eucl':
        return np.sum(np.abs(err)**2)**(1./2)
    
    if ord==np.inf or ord=='oo':
        return np.max(np.abs(err))

    if ord==1 or ord=='1':
        return np.sum(np.abs(err))
    
    raise ValueError('Invalid parameter "ord": {}'.format(ord))


def grav_nbody_calc_diff_eqs(t, y, **kwargs):    
    if 'num_calls' in kwargs:
        kwargs['num_calls'][0] += 1
    
    if 'masses' in kwargs:
        masses = kwargs['masses']
    else:
        raise ValueError('You need to pass masses of points!')

    if 'G' in kwargs:
        G = kwargs['G']
    else:
        G = astr_const.G.cgs.value
    
    has_units = False

    if 'dimension' in kwargs:
        dimension = kwargs['dimension']
    else:
        dimension = 6

    n_bodies = int(len(y) / dimension)
    solved_vec = np.zeros(y.size)
    
    for i in range(n_bodies):
        i_offset = i * dimension
        solved_vec[i_offset] = y[i_offset + 3]
        solved_vec[i_offset + 1] = y[i_offset + 4]
        solved_vec[i_offset + 2] = y[i_offset + 5]
        for j in range(n_bodies):
            j_offset = j * dimension

            if i != j:
                dx = y[i_offset] - y[j_offset]
                dy = y[i_offset + 1] - y[j_offset + 1]
                dz = y[i_offset + 2] - y[j_offset + 2]
                r = (dx**2 + dy**2 + dz**2) ** 0.5
                ax = (-G * masses[j] / r**3) * dx
                ay = (-G * masses[j] / r**3) * dy
                az = (-G * masses[j] / r**3) * dz
                if has_units:
                    ax = ax.value
                    ay = ay.value
                    az = az.value
                solved_vec[i_offset + 3] += ax
                solved_vec[i_offset + 4] += ay
                solved_vec[i_offset + 5] += az
    return solved_vec

def smooth_graph_points(x, factor=0.9):
    smoothed_points = []
    for point in x:
        if smoothed_points:
            prev = smoothed_points[-1]
            smoothed_points.append(prev * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
    

def save_np_array(arr, file_name, delimeter=",", fmt='%.8e', newline='\n', header='', curr_script_dir=False):
    if curr_script_dir:
        np.savetxt(os.path.dirname(os.path.abspath(__file__))+'\\'+file_name, arr, delimiter=delimeter, fmt=fmt, newline=newline, header=header)
    else:
        np.savetxt(file_name, arr, delimiter=delimeter, fmt=fmt, newline=newline, header=header)
