from math import sqrt
import numpy as np


class OdeIntegrator():

    def __init__(self, f=None):
        self.calc_diff_eqs = f
        self.dt = None
        self.t = None
        self.f_params = {}
        self.method_params = {}
        self.y_prev = None
        self.solution_out = None

    def set_solution_out(self, sol_out):
        '''Set callback, which be called at every step.'''
        self.solution_out = sol_out
        return self

    def set_f_params(self, **kwargs):
        '''Set any additional hyperparameters for function f.'''
        self.f_params = kwargs
        return self
    
    def add_f_params(self, **kwargs):
        '''Add any additional hyperparameters for function f.'''
        self.f_params.update(kwargs)
        return self
    
    def set_f(self, f, **kwargs):
        '''
        Assigns an external solver function as the diff-eq solver for method.
        ---------------------------------
        Params:
            f: function which returns a [y] vector for method.
            **kwargs: Any additional hyperparameters.
        '''
        if kwargs:
            self.f_params = kwargs
        self.calc_diff_eqs = f
        return self
    
    def set_method_params(self, **kwargs):
        '''Set any additional hyperparameters for numeric method.'''
        self.method_params = kwargs
        return self
    
    def add_method_params(self, **kwargs):
        '''Add any additional hyperparameters for numeric method.'''
        self.method_params.update(kwargs)
        return self
    
    def set_init_params(self, y, t=0.0, dt=0.001):
        '''Set initial parameters: y, t, dt.'''
        self.dt = dt
        if np.isscalar(y):
            y = [y]
        self.y_prev = np.array(y)
        self.t = t
        return self
    
    def set_integrator(self, method_name, **kwargs):
        '''
        Assigns a numeric method to solve diff-eq
        ---------------------------------
        Params:
            method_name: string name of method.
            **kwargs: Any additional hyperparameters.
        '''        
        if kwargs:
            self.method_params = kwargs
        
        if method_name.upper() == 'RK4':
            self.curr_num_method = self.rk4_method
        elif method_name.upper() == 'RK8':
            self.curr_num_method = self.rk8_method
        elif method_name.upper() == 'PD87' or method_name.upper() == 'DP87':
            self.curr_num_method = self.prince_dormand_87_method
        else:
            raise AttributeError('Not find method {}.'.format(method_name))

    def integrate(self, total_time):
        '''Using the numerical method, find y(total_time).'''
        if not np.isscalar(total_time):
            raise ValueError('Need parametr "t".')
        if not hasattr(self, 'curr_num_method') or not self.curr_num_method:
            raise ValueError('You need to set method. Use function "set_integrator".')      
        if not hasattr(self, 'calc_diff_eqs') or not self.calc_diff_eqs:
            raise ValueError('You need to set f function. Use "set_f".')

        self._check_init_params()

        if total_time < self.t:
            return self.y_prev
        
        if self.dt > (total_time - self.t):
            self.dt = total_time - self.t
        self.t += self.dt
        return self.curr_num_method(total_time)

    def _check_init_params(self):
        if self.y_prev is None or not hasattr(self, 'y_prev'):
            raise ValueError('You need to set start param "y0"!')
        if self.dt is None or not hasattr(self, 'dt'):
            raise ValueError('You need to set start param "dt"!')

    def _call_solution_out(self, t, y):
        ret = 0
        if self.solution_out:
            ret = self.solution_out(t, y)
        return ret

    def rk4_method(self, total_time):
        '''RK4 method. Returns new [y] vector for total_time.'''
        while self.t <= total_time:
            k1 = self.dt * self.calc_diff_eqs(self.t, self.y_prev, **self.f_params) 
            k2 = self.dt * self.calc_diff_eqs(self.t + 0.5*self.dt, self.y_prev + 0.5*k1, **self.f_params)
            k3 = self.dt * self.calc_diff_eqs(self.t + 0.5*self.dt, self.y_prev + 0.5*k2, **self.f_params)
            k4 = self.dt * self.calc_diff_eqs(self.t + self.dt, self.y_prev + k3, **self.f_params)

            y_new = self.y_prev + ((k1 + 2*k2 + 2*k3 + k4) / 6.0)
            self.y_prev = y_new            
            if self._call_solution_out(self.t, y_new) == -1:
                return self.y_prev
            self.t += self.dt
        return self.y_prev
    
    def rk8_method(self, total_time):
        '''RK8 method. Returns new [y] vector for total_time.'''
        while self.t <= total_time:
            k1 = self.calc_diff_eqs(self.t, self.y_prev, **self.f_params)  
            k2 = self.calc_diff_eqs(self.t + self.dt*(4/27), self.y_prev+(self.dt*4/27)*k1, **self.f_params)  
            k3 = self.calc_diff_eqs(self.t + self.dt*(2/9), self.y_prev+(self.dt/18)*(k1 + 3*k2), **self.f_params)  
            k4 = self.calc_diff_eqs(self.t + self.dt*(1/3), self.y_prev+(self.dt/12)*(k1+3*k3), **self.f_params)  
            k5 = self.calc_diff_eqs(self.t + self.dt*(1/2), self.y_prev+(self.dt/8)*(k1+3*k4), **self.f_params)  
            k6 = self.calc_diff_eqs(self.t + self.dt*(2/3), self.y_prev+(self.dt/54)*(13*k1-27*k3+42*k4+8*k5), **self.f_params)  
            k7 = self.calc_diff_eqs(self.t + self.dt*(1/6), self.y_prev+(self.dt/4320)*(389*k1-54*k3+966*k4-824*k5+243*k6), **self.f_params)  
            k8 = self.calc_diff_eqs(self.t + self.dt, self.y_prev+(self.dt/20)*(-231*k1+81*k3-1164*k4+656*k5-122*k6+800*k7), **self.f_params) 
            k9 = self.calc_diff_eqs(self.t + self.dt*(5/6), self.y_prev+(self.dt/288)*(-127*k1+18*k3-678*k4+456*k5-9*k6+576*k7+4*k8), **self.f_params) 
            k10 = self.calc_diff_eqs(self.t + self.dt, self.y_prev+(self.dt/820)*(1481*k1-81*k3+7104*k4-3376*k5+72*k6-5040*k7-60*k8+720*k9), **self.f_params)  

            y_new = self.y_prev + self.dt/840*(41*k1+27*k4+272*k5+27*k6+216*k7+216*k9+41*k10)
            self.y_prev = y_new
            if self._call_solution_out(self.t, y_new) == -1:
                return self.y_prev
            self.t += self.dt
        return self.y_prev

    def prince_dormand_87_method(self, total_time):
        '''
        DOPRI 8(7) method. Explicit runge-kutta method with stepsize control.
        Method accepts the following hyperparameters:
            - atol : absolute tolerance for solution
            - rtol : relative tolerance for solution
            - mitig_param: "softening" factor on new step selection
            - ord: order of the norm (type of norm) in calc_err_norm().
            - calc_tol: can be set by the user
            - calc_tol_params: hyperparameters for calc_tol()
        '''
        c2=1/18;                   a21=1/18
        c3=1/12;                   a31=1/48;                    a32=1/16
        c4=1/8;                    a41=1/32;                    a43=3/32
        c5=5/16;                   a51=5/16;                    a53=-75/64;                     a54=75/64
        c6=3/8;                    a61=3/80;                    a64=3/16;                       a65=3/20
        c7=59/400;                 a71=29443841/614563906;      a74=77736538/692538347;         a75=-28693883/1125000000;    a76=23124283/1800000000
        c8=93/200;                 a81=16016141/946692911;      a84=61564180/158732637;         a85=22789713/633445777;      a86=545815736/2771057229;      a87=-180193667/1043307555
        c9=5490023248/9719169821;  a91=39632708/573591083;      a94=-433636366/683701615;       a95=-421739975/2616292301;   a96=100302831/723423059;       a97=790204164/839813087;       a98=800635310/3783071287
        c10=13/20;                 a10_1=246121993/1340847787;  a10_4=-37695042795/15268766246; a10_5=-309121744/1061227803; a10_6=-12992083/490766935;     a10_7=6005943493/2108947869;   a10_8=393006217/1396673457;    a10_9=123872331/1001029789
        c11=1201146811/1299019798; a11_1=-1028468189/846180014; a11_4=8478235783/508512852;     a11_5=1311729495/1432422823; a11_6=-10304129995/1701304382; a11_7=-48777925059/3047939560; a11_8=15336726248/1032824649;  a11_9=-45442868181/3398467696; a11_10=3065993473/597172653
        c12=1.0;                   a12_1=185892177/718116043;   a12_4=-3185094517/667107341;    a12_5=-477755414/1098053517; a12_6=-703635378/230739211;    a12_7=5731566787/1027545527;   a12_8=5232866602/850066563;    a12_9=-4093664535/808688257;   a12_10=3962137247/1805957418; a12_11=65686358/487910083
        c13=1.0;                   a13_1=403863854/491063109;   a13_4=-5068492393/434740067;    a13_5=-411421997/543043805;  a13_6=652783627/914296604;     a13_7=11173962825/925320556;   a13_8=-13158990841/6184727034; a13_9=3936647629/1978049680;   a13_10=-160528059/685178525;  a13_11=248638103/1413531060

        b1=13451932/455176623; b6=-808719846/976000145; b7=1757004468/5645159321; b8=656045339/265891186; b9=-3867574721/1518517206; b10=465885868/322736535; b11=53011238/667516719; b12=2/45
        b_1=14005451/335480064; b_6=-59238493/1068277825; b_7=181606767/758867731; b_8=561292985/797845732; b_9=-1041891430/1371343529; b_10=760417239/1151165299; b_11=118820643/751138087; b_12=-528747749/2220607170; b_13=1/4


        adaptive = self.method_params['adapt'] if 'adapt' in self.method_params else False

        if adaptive:
            if 'atol' in self.method_params:
                a_tol = self.method_params['atol']
            else:
                raise AttributeError('You need to set "atol" param for PD8(7) method!')
                
            if 'rtol' in self.method_params:
                r_tol = self.method_params['rtol']
            else:
                raise AttributeError('You need to set "rtol" param for PD8(7) method!')

            if 'mitig_param' in self.method_params:
                mitig_param = self.method_params['mitig_param']
            else:
                mitig_param = 1.0

            if 'ord' in self.method_params:
                ord = self.method_params['ord']
            else:
                ord = None  
            
            if 'ifactor' in self.method_params:
                ifactor = self.method_params['ifactor']
            else:
                ifactor = 10.0            
            if 'dfactor' in self.method_params:
                dfactor = self.method_params['dfactor']
            else:
                dfactor = 10.0

            if 'nsteps' in self.method_params:
                nsteps = self.method_params['nsteps'] 
            else:
                nsteps = 1e3      
            
            if 'calc_tol' in self.method_params:
                calc_tol = self.method_params['calc_tol']
                calc_tol_kwargs = self.method_params['calc_tol_params'] if 'calc_tol_params' in self.method_params else ()
            else:
                calc_tol = calc_tol_            
        
        _nsteps = 0

        while self.t <= total_time:
            k1 = self.calc_diff_eqs(self.t, self.y_prev, **self.f_params)
            k2 = self.calc_diff_eqs(self.t+c2*self.dt, self.y_prev+self.dt*(a21*k1), **self.f_params)
            k3 = self.calc_diff_eqs(self.t+c3*self.dt, self.y_prev+self.dt*(a31*k1+a32*k2), **self.f_params)
            k4 = self.calc_diff_eqs(self.t+c4*self.dt, self.y_prev+self.dt*(a41*k1+a43*k3), **self.f_params)
            k5 = self.calc_diff_eqs(self.t+c5*self.dt, self.y_prev+self.dt*(a51*k1+a53*k3+a54*k4), **self.f_params)
            k6 = self.calc_diff_eqs(self.t+c6*self.dt, self.y_prev+self.dt*(a61*k1+a64*k4+a65*k5), **self.f_params)
            k7 = self.calc_diff_eqs(self.t+c7*self.dt, self.y_prev+self.dt*(a71*k1+a74*k4+a75*k5+a76*k6), **self.f_params)
            k8 = self.calc_diff_eqs(self.t+c8*self.dt, self.y_prev+self.dt*(a81*k1+a84*k4+a85*k5+a86*k6+a87*k7), **self.f_params)
            k9 = self.calc_diff_eqs(self.t+c9*self.dt, self.y_prev+self.dt*(a91*k1+a94*k4+a95*k5+a96*k6+a97*k7+a98*k8), **self.f_params)
            k10 = self.calc_diff_eqs(self.t+c10*self.dt, self.y_prev+self.dt*(a10_1*k1+a10_4*k4+a10_5*k5+a10_6*k6+a10_7*k7+a10_8*k8+a10_9*k9), **self.f_params)
            k11 = self.calc_diff_eqs(self.t+c11*self.dt, self.y_prev+self.dt*(a11_1*k1+a11_4*k4+a11_5*k5+a11_6*k6+a11_7*k7+a11_8*k8+a11_9*k9+a11_10*k10), **self.f_params)
            k12 = self.calc_diff_eqs(self.t+c12*self.dt, self.y_prev+self.dt*(a12_1*k1+a12_4*k4+a12_5*k5+a12_6*k6+a12_7*k7+a12_8*k8+a12_9*k9+a12_10*k10+a12_11*k11), **self.f_params)
            k13 = self.calc_diff_eqs(self.t+c13*self.dt, self.y_prev+self.dt*(a13_1*k1+a13_4*k4+a13_5*k5+a13_6*k6+a13_7*k7+a13_8*k8+a13_9*k9+a13_10*k10+a13_11*k11), **self.f_params)

            y_new = self.y_prev + self.dt*(b1*k1+b6*k6+b7*k7+b8*k8+b9*k9+b10*k10+b11*k11+b12*k12)
            y_new_ = self.y_prev + self.dt*(b_1*k1+b_6*k6+b_7*k7+b_8*k8+b_9*k9+b_10*k10+b_11*k11+b_12*k12+b_13*k13)

            if adaptive:
                tol = calc_tol(y_new, y_new_, a_tol, r_tol, **calc_tol_kwargs)
                err = calc_err_(tol, y_new, y_new_)
                err_n = calc_err_norm(err, ord)
                prev_t = self.dt

                if err_n == 0.0:
                    err_n = 1e-6
                self.dt = mitig_param * self.dt * (1/err_n)**(1/8)

                if self.dt/prev_t > ifactor:
                    self.dt = prev_t * ifactor
                if prev_t/self.dt > dfactor:
                    self.dt = prev_t/dfactor

                _nsteps += 1
                if _nsteps > nsteps:
                    raise RuntimeError('Limit exceeded of nsteps.')

                if err_n <= 1.000:
                    _nsteps = 0
                    self.t += self.dt
                    self.y_prev = y_new_
                    if self._call_solution_out(self.t, y_new) == -1:
                        return self.y_prev            
            else:
                self.t += self.dt
                self.y_prev = y_new_
                if self._call_solution_out(self.t, y_new) == -1:
                    return self.y_prev
        return self.y_prev