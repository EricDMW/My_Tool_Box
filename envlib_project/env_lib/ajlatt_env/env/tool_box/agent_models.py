"""Dynamic Object Models

The following models are included:

    AgentDoubleInt2D : Double Integrator Model in 2D
                        state: x,y,xdot,ydot
    AgentDoubleInt2D_Nonlinear : Double Integrator Model with non-linear term for obstalce avoidance in 2D
                        state: x,y,xdot,ydot
    AgentSE2 : SE2 Model
           state x,y,theta

    Agent2DFixedPath : Model with a pre-defined path
"""
# Solve Potential Path Issues
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent
sys.path.append(str(parent_dir))

import numpy as np
import util
from scipy.linalg import block_diag
from numpy import cos, sin,eye,ones,zeros

class Agent(object):
    def __init__(self, dim, sampling_period):
        self.dim = dim
        self.sampling_period = sampling_period

    def reset(self, init_state):
        self.state = init_state
    
    def propagate(self, control_input=None):
        """
        True position propagate
        Parameters:
        ----------
        control_input : list. [linear_velocity, angular_velocity]

        """

        new_state = SE2Dynamics(self.state, self.sampling_period, control_input)                
        # print('propgated new state = ',new_state)
        self.state = new_state

class Agent_est(Agent):
    def __init__(self, dim, sampling_period):
        super().__init__(dim, sampling_period)
    
    def reset(self, init_state,init_cov):
        # init_est_state = np.random.normal(init_state, init_cov.diagonal())
        init_est_state = np.random.normal(init_state, np.sqrt(init_cov.diagonal()))
        self.state = init_est_state
        self.cov = init_cov

    
    def propagate(self, control_input,sigma_vT,sigma_wT ):
        J = np.array([[0, -1],[1, 0]])

        # Q = diag([  sigma_vT**2,sigma_wT**2  ]);
        Q = block_diag(sigma_vT**2, sigma_wT**2)

        # Propagate Target State

        new_state = SE2Dynamics(self.state, self.sampling_period, control_input)     
        # a = eye(2)
        # b = J@(new_state[0:2]-self.state[0:2]).reshape(2,1) 
        # c = zeros((1,2))
        # d = 1
        # Propagate Covariance
        # PHI = np.block([[ eye(2) ,      J@(new_state[0:2]-self.state[0:2] ) ], 
        #                 [zeros((1,2)),  1           ]])

        PHI = np.block([
                        [eye(2),  J@(new_state[0:2]-self.state[0:2]).reshape(2,1)  ], 
                        [zeros((1,2)),  1 ]])

        # PHI = [ eye(2)      J@( new_state[0:2]-self.state[0:2] ) ;
        #         zeros(1,2)  1 ]
        dt = self.sampling_period

        G = np.block([   
            [dt*cos(self.state[2]),     0],
            [dt*sin(self.state[2] ),    0],
            [0,                         dt]])
        # G = [G_R zeros(3,2); zeros(3,2) G_T];
        Qprime = G@Q@G.T
        Pe = PHI@self.cov@PHI.T + Qprime
        # Pe = PHI@self.cov@PHI.T
        Pe = 0.5*(Pe+Pe.T)

        self.state  = new_state
        self.cov    = Pe
        
    

def SE2Dynamics(x, dt, u):
    """
    update dynamics function with a control input -- linear, angular velocities
    """
    # u1 = u.squeeze().astype('float64')
    assert(len(x)==3)
    # tw = dt * u[1]
    # diff = np.zeros((3,1))
    # Propagate the agent state
    # diff = np.array([u[0]/u[1]*(np.sin(x[2]+tw)-np.sin(x[2])),
    #                 u[0]/u[1]*(np.cos(x[2]) - np.cos(x[2]+tw)),
    #                 tw])

    diff = np.array([
        u[0]*dt*cos(x[2]),
        u[0]*dt*sin(x[2]),
        dt * u[1]])

    new_x = x + diff 
    new_x[2] = util.pi_to_pi(new_x[2])


    # x_true(1:3) = [  u[0]*dt*cos(x[2]) ;
    #  v_m*dt*sin(x_true(3));
    # omega_true*dt ];
    
    return new_x



def SE2Dynamics_V2(x, dt, u):
    """
    update dynamics function with a control input -- linear, angular velocities
    """
    u1 = u.squeeze().astype('float64')
    assert(len(x)==3)
    tw = dt * u[1]
    # diff = np.zeros((3,1))
    # Propagate the agent state
    if abs(tw) < 0.001:
        diff = np.array([dt*u[0]*np.cos(x[2]+tw/2), 
                        dt*u[0]*np.sin(x[2]+tw/2),
                        tw])
    else:
        diff = np.array([u[0]/u[1]*(np.sin(x[2]+tw)-np.sin(x[2])),
                        u[0]/u[1]*(np.cos(x[2]) - np.cos(x[2]+tw)),
                        tw])
    new_x = x + diff
    new_x[2] = util.pi_to_pi(new_x[2])
    
    return new_x

