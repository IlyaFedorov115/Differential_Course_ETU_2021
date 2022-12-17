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

from body import *
from functions import *
from odeintegrator import *

class Simulation():
    '''
    Simulation object.
    ------------------
    Params:
        bodies: list of Body() objects.
        has_units: type of bodies dimension.
        num_bodies: number of particles.
        _n_dim: len(y) for 'y' vector for 1 particle.
        quant_vec: vector of 'y' vectors for every body.
        mass_vec: vector of masses for all bodies.
        name_vec: vector of names for all bodies.
    '''
    def __init__(self, bodies, has_units=None):
        self.has_units = has_units
        self.bodies = bodies
        self.num_bodies = len(self.bodies)
        self._n_dim = 6
        self.quant_vec = np.concatenate(np.array([elem.return_vec() for elem in self.bodies]))
        self.mass_vec = np.array([elem.return_mass() for elem in self.bodies])
        self.name_vec = [elem.return_name() for elem in self.bodies]
        self.system_mass = np.sum(self.mass_vec)
        self.ode_integrator = OdeIntegrator()

    def set_ode_integator(self, integrator):
        if not isinstance(integrator, OdeIntegrator):
            raise ValueError('Error type of integrator.')
        self.ode_integrator = integrator
        return self

    def set_diff_eqs(self, calc_diff_eqs, **kwargs):
        self.ode_integrator.set_f(calc_diff_eqs, **kwargs)
    
    def set_numeric_method(self, method_name, **kwargs):
        self.ode_integrator.set_integrator(method_name, **kwargs)     
  
    def run_simulation(self, total_time, dt, t_0=0, G=None, logging=False):
        '''
        Method for running simulation.
        ------------------------------
        Params:
            total_time: total time for simulation.
            dt: timestep.
            t_0: start time.
        '''

        if G is None:
            if self.has_units.upper() == 'CGS':
                G = astr_const.G.cgs.value
            elif self.has_units.upper() == 'SI':
                G = astr_const.G.si.value
            else:
                raise TypeError('You need to set value of "G" if you use dimensionless system.')
        else:
            if self.has_units:
                try:
                    _ = G.unit
                except:
                    G = (G * astr_const.G.unit)
                G = G.cgs.value if self.has_units.upper() == 'CGS' else G.si.value
        self.G = G

        if self.has_units:
            try:
                _ = t_0.unit
            except:
                t_0 = (total_time.unit * t_0)#.cgs.value
            if self.has_units.upper() == 'CGS':
                t_0 = t_0.cgs.value
                total_time = total_time.cgs.value
                dt = dt.cgs.value
            elif self.has_units.upper() == 'SI':
                t_0 = t_0.si.value
                total_time = total_time.si.value
                dt = dt.si.value
            else:
                raise TypeError('Unintended unit system type {}!'.format(self.has_units))

        self.num_diff_eq_calls = 0
        self.quant_vec = np.concatenate(np.array([elem.return_vec() for elem in self.bodies]))
        self.history = [self.quant_vec]
        self.ode_integrator.add_f_params(G=G, num_calls=[0], masses=self.mass_vec)
        self.ode_integrator.set_init_params(self.quant_vec, t_0, dt)
        self.ode_integrator.add_method_params(calc_tol=calc_tol_n, calc_tol_params={'nbody':self.num_bodies})
        def solout(t,y):
            self.history.append(y)
            self.quant_vec = y
        self.ode_integrator.set_solution_out(solout)
        start_time = time.time()

        self.ode_integrator.integrate(total_time)

        end_time = time.time() - start_time
        print('Simulation passed in {} seconds'.format(end_time))
        self.history = np.array(self.history)
        self.num_diff_eq_calls = self.ode_integrator.f_params['num_calls'][0]
    
    def get_num_calls_diff_eq(self):
        if hasattr(self, 'num_diff_eq_calls'):
            return self.num_diff_eq_calls
        else:
            return None

    def _calc_barycent_v(self, index):
        barycent_v = np.zeros(3)
        for i in range(self.num_bodies):
            offset = i * 6
            barycent_v += self.mass_vec[i] * self.history[index][offset+3:offset+6]
        return barycent_v / self.system_mass

    def plot_barycent_v_dist_history(self, barycent_v0, barycent_v_history, need_save_plt=False, save_plt_name=None, smooth=True):
        dist_hist = ([np.linalg.norm(barycent_v0-barycent_v_history[i]) for i in range(len(barycent_v_history))])
        plt.axhline(y=0, xmin=0, xmax=len(barycent_v_history), color ="green")
        if smooth:
            plt.plot(np.arange(0, len(barycent_v_history), 1), smooth_graph_points(dist_hist))
        else:
            plt.plot(np.arange(0, len(barycent_v_history), 1), dist_hist)
        plt.grid()
        if need_save_plt:
            if save_plt_name:
                plt.savefig(save_plt_name)
            else:
                now = datetime.datetime.now()
                plt.savefig(r"barycent_history"+now.strftime("%d-%m-%Y_%H-%M-%S")+".png")
        plt.show()

    def check_barycent_inv(self, need_plot=True, need_save_plt=False, save_plt_name=None, smooth=True):
        if not hasattr(self, 'history'):
            raise AttributeError('Missing attribute "history", maybe need run simulation?')
        barycent_v0 = self._calc_barycent_v(0)
        barycent_v_history = []
        for i in range(len(self.history)):
            barycent_v_history.append(self._calc_barycent_v(i))      
        if need_plot:
            self.plot_barycent_v_dist_history(barycent_v0, np.array(barycent_v_history), need_save_plt, save_plt_name, smooth)
        return np.array(barycent_v_history)


    def _calc_barycent_r(self, index):
        barycent_r = np.zeros(3)
        for i in range(self.num_bodies):
            offset = i * 6
            barycent_r += self.mass_vec[i] * self.history[index][offset:offset+3]
        return barycent_r / self.system_mass

    def get_barycent_r_history(self):
        if not hasattr(self, 'history'):
            raise AttributeError('Missing attribute "history", maybe need run simulation?')
        barycent_r_history = []
        for i in range(len(self.history)):
            barycent_r_history.append(self._calc_barycent_r(i))
        return np.array(barycent_r_history)
    

    def check_energy_inv(self, need_smooth=True, factor=0.5, title_pad=20):
        if not hasattr(self, 'history'):
            raise AttributeError('Missing attribute "history", maybe need run simulation?')
        T_history, U_history, E_history = [], [], []
        for i in range(len(self.history)):
            T_history.append(self._calc_k_energy(self.history[i]))
            U_history.append(self._calc_p_energy(self.history[i]))
            E_history.append(T_history[i] - U_history[i])
        E_history = np.array(E_history)

        E0 = E_history[0]
        rel_ch = np.array(np.abs(E_history-E0)/np.abs(E0))

        if need_smooth:
            E_history = smooth_graph_points(E_history, factor=factor)
            rel_ch = smooth_graph_points(rel_ch, factor=factor)
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(12, 6))
        #ax1.axhline(y=E0, xmin=0, xmax=len(self.history), color ="green")
        ax1.axhline(y=E0, xmin=0, xmax=len(self.history), color ="green", label='$E_0$')
        ax1.plot(range(0, len(self.history), 1), E_history, color="blue", linestyle='-', linewidth=1)
        ax1.set_ylabel(r'$E$')
        ax1.set_title('Total system energy history.', pad=title_pad)
        ax1.legend(loc="upper right")

        ax2.plot(range(0, len(self.history), 1), rel_ch, color="blue", linestyle='-', linewidth=1)
        ax2.set_ylabel(r'$\frac{dE}{E_0}$')
        ax2.set_title('Relative energy changes', pad=title_pad)
        plt.show()
        return E_history, E0, rel_ch[-1]


    def _calc_k_energy(self, y_vec):
        res = []
        for i in range(self.num_bodies):
            offset = i * 6
            v_vec = y_vec[offset+3:offset+6]
            res.append(self.mass_vec[i] * np.dot(v_vec,v_vec))
        res = kahan_sum(np.array(res))
        return res / 2.0
    
    def _calc_p_energy(self, y_vec):
        res_p = []
        for i in range(self.num_bodies):
            ioffset = i * 6
            res = 0
            for j in range(self.num_bodies):
                joffset = j * 6
                if i != j:
                    dx = y_vec[ioffset] - y_vec[joffset]
                    dy = y_vec[ioffset+1] - y_vec[joffset+1]
                    dz = y_vec[ioffset+2] - y_vec[joffset+2]
                    r = (dx**2+dy**2+dz**2)**0.5
                    res += self.mass_vec[i] * self.mass_vec[j] / r
            res_p.append(res)
        res_p = kahan_sum(np.array(res_p))
        return res_p * self.G / 2.0
        

    def plot(self, point_size = 30, orbit_width = 1, 
            orbit_alpha = 1, point_alpha = 1, point_marker = 'o', 
            linestyle = '-', point_edge_width=1):
        if not hasattr(self, 'history'):
            raise AttributeError('Missing attribute "history", maybe need run simulation?')
        fig = plt.figure(figsize=(11,5))
        fig.suptitle('Graphical results',fontsize=14, fontstyle='italic')
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        if self.has_units and self.has_units.upper() == 'CGS':
            unit = 'cm'
        elif self.has_units and self.has_units.upper() == 'SI':
            unit = 'm'
        else:
            unit = 'unit'
        ax1.set_xlabel(unit)
        ax1.set_ylabel(unit) 
        for i in range(len(self.bodies)):
            offset = i * 6
            x_ = self.history[0][offset + 0]
            y_ = self.history[0][offset + 1]
            z_ = self.history[0][offset + 2]
            x, y, z = [], [], []
            colors_1 = np.random.rand(3)
            colors_2 = np.random.rand(3)
            ax1.scatter(x=x_, y=y_, marker=point_marker, color=colors_1, 
                        linewidths=point_edge_width, edgecolor=colors_2, alpha=point_alpha, s=point_size)
            ax2.scatter(x_, y_, z_, marker=point_marker, color=colors_1, alpha=point_alpha, s=point_size)
            for j in range(len(self.history)):
                x.append(self.history[j][offset])
                y.append(self.history[j][offset + 1])
                z.append(self.history[j][offset + 2])
            ax1.plot(x,y, color=colors_1, 
                    linewidth=orbit_width, alpha=orbit_alpha,
                    label='{}'.format(self.name_vec[i]),
                    linestyle=linestyle)
            ax2.plot(x,y,z, color=colors_1,)
        ax1.grid(color='black', linewidth=0.3, linestyle='--')
        ax1.legend(fontsize=6, ncol=2, facecolor='oldlace',
                   edgecolor='green', title='Orbits', title_fontsize='8', loc='lower right')
        plt.show()


    def log_simulation(self, step, n_steps, clock_time):
        '''Method to logging simulation in console??'''
        sys.stdout.flush()
        sys.stdout.write('Integrating: step = {} / {} | simulation time = {}'.format(step, n_steps, round(clock_time,3)) + '\r')

    def print_unit_system(self):
        if self.has_units.upper() == 'CGS' or self.has_units.upper() == 'SI':
            print(' "{}" unit system'.format(self.has_units.upper()))
        else:
            print('Dimensionless system.')