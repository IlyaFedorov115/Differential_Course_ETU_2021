from simulation import *
from body import *
from functions import *
from odeintegrator import *

'''
COURSE WORK 8383
Fedorov, Grechko
'''


if __name__ == '__main__':
    
    Sun = Body(mass=astr_const.M_sun.cgs,
            x_vec = np.array([0,0,0])*astr_un.km,
            v_vec = np.array([0,0,0])*astr_un.km/astr_un.s,
            name='Sun',
            has_units='cgs')  

    Earth = Body(mass=astr_const.M_earth.cgs,
            x_vec = np.array([astr_un.au.si.to(astr_un.km),0,0])*astr_un.km,
            v_vec = np.array([0,29.022,0])*astr_un.km/astr_un.s,
            name='Earth',
            has_units='cgs')  

    Mercury = Body(mass=(0.33022e24*astr_un.kg),
            x_vec = np.array([0.38710 * astr_un.au.si.to(astr_un.km),0,0])*astr_un.km,
            v_vec = np.array([0,-47.36,0])*astr_un.km/astr_un.s,
            name='Mercury',
            has_units='cgs')  

    Venus = Body(mass=(4.8690e24*astr_un.kg),
            x_vec = np.array([-0.72333 * astr_un.au.si.to(astr_un.km),0,0])*astr_un.km,
            v_vec = np.array([0,35.02,0])*astr_un.km/astr_un.s,
            name='Venus',
            has_units='cgs')

    Mars = Body(mass=(0.64191e24*astr_un.kg),
            x_vec = np.array([1.52363 * astr_un.au.si.to(astr_un.km),0,0])*astr_un.km,
            v_vec = np.array([0,-24.077,0])*astr_un.km/astr_un.s,
            name='Mars',
            has_units='cgs')

    Jupiter = Body(mass=(1898.8e24*astr_un.kg),
            x_vec = np.array([-5.20441 * astr_un.au.si.to(astr_un.km),0,0])*astr_un.km,
            v_vec = np.array([0,13.07,0])*astr_un.km/astr_un.s,
            name='Jupiter',
            has_units='cgs')

    Saturn = Body(mass=(568.50e24*astr_un.kg),
            x_vec = np.array([9.58378 * astr_un.au.si.to(astr_un.km),0,0])*astr_un.km,
            v_vec = np.array([0,-9.69,0])*astr_un.km/astr_un.s,
            name='Saturn',
            has_units='cgs')


#####
    bodies = [Earth, Sun,  Mercury, Venus, Mars, Jupiter, Saturn]

    ''' Run and plot '''
    simulation = Simulation(bodies, has_units='cgs')
    simulation.set_diff_eqs(grav_nbody_calc_diff_eqs)
    #simulation.set_numeric_method("DP87", adapt=True, atol=np.array([0.001, 0.0000001]), rtol=np.array([0.00001, 0.0000000001]), ord=None, mitig_param=0.7, ifactor=10, dfactor=10)
    #simulation.set_numeric_method("DP87", adapt=False)
    #simulation.set_numeric_method("RK8")
    simulation.set_numeric_method("RK4")
    simulation.run_simulation(4*astr_un.year, 12*astr_un.h)
    print(simulation.get_num_calls_diff_eq())
    simulation.check_barycent_inv(smooth=True)
    print(simulation.check_energy_inv(factor=0.6)[1:3])

    simulation.plot()


