#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : ajlatt_agent.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: RL_LLM
@Date   : 10/05/2024
@Time   : 18:56:07
@Info   : Definition of Agents in ajlatt senerio
"""
# Solve Potential Path Issues
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parents[0]
sys.path.append(str(parent_dir))


import gym
import tool_box.agent_models
import os
import copy
import torch


import numpy as np
import tool_box.util as util
import matplotlib.pyplot as plt


from gym import wrappers, spaces, logger
from numpy.linalg import det, inv, norm,matrix_rank,eig
from numpy import cos, sin, pi, sqrt, ones, zeros, trace
from tool_box.agent_models import *
from math import inf
from tool_box.util import cartesian2polar, pi_to_pi, rotation_matrix
from scipy.linalg import block_diag, pinv
from scipy.optimize import minimize
from tool_box.display_wrapper import Display2D

from maps import map_utils
from maps.dynamic_map import DynamicMap
from tool_box import Target_Moving
from statistics import mean
from parameters import all_parsers




def make(figID=0, *args, **kwargs):
    """
    Parameters:
    ----------
    env_name : str
        name of an environment. (e.g. 'TargetTracking-v0')
    render : bool
        wether to render.
    figID : int
        figure ID for rendering and/or recording.
    record : bool
        whether to record a video.
    ros : bool
        whether to use ROS.
    directory :str
        a path to store a video file if record is True.
    T_steps : int
        the number of steps per episode.
    num_target : int
        the number of targets
    """
    parser = all_parsers.parse_args()
    # Override parser attributes if present in kwargs
    for key, value in kwargs.items():
        if hasattr(parser, key):
            setattr(parser, key, value)

    # T_steps = 100
    print(f'----------------------------------Current Making Env {parser.env_name}-----------------------------------')
    

    
    local_view = 0
    env0 = TargetTrackingEnv0(parser, **kwargs)
    env = wrappers.TimeLimit(env0, max_episode_steps=parser.T_steps)
    # from ttenv.display_wrapper import Display2D
    # env = Display2D(env, figID=figID, local_view=local_view)

    if parser.render:
        env = Display2D(env, figID=figID, local_view=local_view)

    return env


# For CI usage
global s_temp
s_temp = []


class TargetTrackingEnv0(gym.Env):
    def __init__(self, parser, is_training=True,
                 known_noise=True, **kwargs):
        gym.Env.__init__(self)
        
        # parse the parameters used in the following functions and varialbles 
        self.parser = parser
        self.map_name = parser.map_name
        # maximum running lenth despide of any conditions
        self.maxmum_run = parser.maxmum_run
        
        # for desired target trajectories
        Target_Moving.Choose_map(parser)
        self.target_controller = Target_Moving.target_action
        
        
        if parser.fix_seed:
            self.seed(parser.seed)

        self.sampling_period = parser.sampling_period # sec
        
        
        self.action_space = spaces.Box(low=np.array(
            [0.0, -pi/4]), 
            high=np.array([2.0, pi/4]), shape=(2,),dtype=np.float64)   
         
        self.action_dim = self.action_space.shape[0]

        self.state_num = 7+4*(parser.num_Robot-1)+3+2
        self.obs_dim = self.state_num
        self.observation_space = spaces.Box(low=-np.inf*ones( self.state_num ), 
            high=np.inf*ones(self.state_num ), shape=(self.state_num,), dtype=np.float64)



        self.nR = parser.num_Robot  # Assuming num_Robot is equivalent to nb_targets
        self.nT = parser.num_targets   # Assuming num_Target can also be set to nb_targets

        # noise is the percentage of distance measurement
        self.sigma_p = parser.sigma_p
        self.sigma_r = parser.sigma_r  # sensor range noise.
        self.sigma_th = parser.sigma_th  # sensor bearing noise

        self.SIGPERCENT = parser.SIGPERCENT
        self.useupdate = parser.useupdate

        # self.viewer = None

        self.T_steps = parser.T_steps

        self.min_range = parser.sensor_r_min
        self.max_range = parser.sensor_r_max
        self.max_com_range = parser.commu_r_max
        self.fov = parser.fov

        self.robot_init_pos = parser.robot_init_pose


        self.robot_est_init_pos = self.robot_init_pos

        self.robot_init_cov = np.zeros((parser.num_Robot,3,3))
        for i in range(parser.num_Robot):
            self.robot_init_cov[i,:,:] = parser.robot_init_cov[i]*np.identity(3)
            self.robot_init_cov[i,2,2] = 1e-3

        self.robot_linear_velocity = 0

        self.sigmapR = parser.sigmapR
        self.sigmaR = self.sigmapR*self.robot_linear_velocity
        self.sigma_vR = self.sigmaR/sqrt(2)
        self.sigma_wR = 2*sqrt(2)*self.sigmaR


        self.target_init_pos = parser.target_init_pose
        self.target_est_init_pos = self.target_init_pos


        # target_init_cov  = [1]+ [1e-1]*self.nT
        target_init_cov  = parser.target_init_cov

        self.target_init_cov = np.zeros((parser.num_Target,3,3))
        for i in range(parser.num_Target):
            self.target_init_cov[i,:,:] = target_init_cov[i]*np.identity(3)
            self.target_init_cov[i,2,2] = 1e-1

        self.target_linear_velocity = parser.target_linear_velocity
        # Set to store the initial state
        self.target_angular_velocity = 0.2

        self.target_omega_bound = parser.target_omega_bound

        map_dir_path = '/'.join(map_utils.__file__.split('/')[:-1])
        if 'dynamic_map' in self.map_name :
            self.MAP = DynamicMap(
                map_dir_path = map_dir_path,
                map_name = self.map_name,
                margin2wall = parser.margin2wall)
        else:
            try:
                self.MAP = map_utils.GridMap(
                    map_path=os.path.join(map_dir_path, self.map_name),
                    margin2wall = parser.margin2wall)
            except Exception as e:
                print(f"Warning: Failed to load map '{self.map_name}': {e}")
                print("Falling back to 'empty' map")
                self.map_name = "empty"
                self.MAP = map_utils.GridMap(
                    map_path=os.path.join(map_dir_path, self.map_name),
                    margin2wall = parser.margin2wall)


        # fix the processing noise or not
        if parser.process_noise_fixed:
            self.sigma_vT = parser.sigma_vT
            self.sigma_wT = parser.sigma_wT
        else:
            self.sigmapT = parser.sigmapT
            self.sigmaT = self.sigmapT*self.target_linear_velocity
            self.sigma_vT = self.sigmaT/sqrt(2)
            self.sigma_wT = 2*sqrt(2)*self.sigmaT

        self.num_agent = parser.num_Robot
        
        # Solve the confliction of env and ppo parts
        self.agent_num = self.num_agent
        
        self.num_target = parser.num_Target
        self.episode_num = parser.episode_num

        self.observe_list = []

        # self.robot_rmse = [
        #     [[] for i in range(self.num_agent)] for j in range(self.episode_num)]
        # self.target_rmse = [
        #     [[] for i in range(self.num_agent)] for j in range(self.episode_num)]

        # self.robot_cov_trace_list = [
        #     [[] for i in range(self.num_agent)] for j in range(self.episode_num)]
        # self.target_cov_trace_list = [
        #     [[] for i in range(self.num_agent)] for j in range(self.episode_num)]
        
        # record the corresponding information while keep the lenth of each row be flexible
        self.record_unit = [[] for i in range(self.num_agent)]
        
        

        self.robot_cov_trace_list = []
        self.robot_cov_trace_list.append(self.record_unit)
        self.target_cov_trace_list = []
        self.target_cov_trace_list.append(self.record_unit)
        

        # # For CI usage
        # self.s_temp = []

        # Build an agent, targets, and beliefs.
        self.build_models()
        self.i_episode = -1

        self.com_plot = np.zeros((self.nR,self.nR))
        self.RT_obs = np.zeros((self.nR, self.nR + self.nT), dtype=int)
        self.end_count = 0
        self.time_step_count = 0
        self.unobserved_count = 0
        self.observed_count = 0
        
        
        # Record the number of collisions
        self.obstacles_count = []
        self.obstacles_count_unit = [0 for _ in range(self.num_agent)]
        self.obstacles_count.append(self.obstacles_count_unit)
        # self.obstacles_count = [
            # [ 0 for i in range(self.num_agent)] for j in range(self.episode_num)]

        self.state = zeros((self.num_agent, self.state_num ))

    def seed(self, seed_num):
        # Set the seed for NumPy's random number generator
        np.random.seed(seed_num)
        
        # Set the seed for PyTorch's random number generator
        torch.manual_seed(seed_num)
        
        # If you're using CUDA, set the seed for CUDA's RNG as well
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_num)
            torch.cuda.manual_seed_all(seed_num)  # If using multi-GPU setups

        # Optional: Make operations deterministic (might slow down performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    
    def reset(self):
        
        # Truncated when agents run over given time steps
        self.truncated_indice = 0
        self.truncated = False
        
        # Initialize the episode
        self.i_episode = self.i_episode + 1
        
        for i in range(self.num_target):
            self.target_true[i].reset(self.target_init_pos[i])


        for i in range(self.num_agent):                    
            self.robot_est_init_pos[i] = self.robot_init_pos[i]

            self.robot_true[i].reset(self.robot_init_pos[i])

            self.robot_est[i].reset(
                    self.robot_est_init_pos[i], self.robot_init_cov[i])


            for idx_T in range(self.num_target):
                # self.target_est[i][idx_T].reset(
                #     self.target_init_pos[idx_T,:], self.target_init_cov)
                if idx_T==0:
                    self.target_est[i][idx_T].reset(
                        self.target_est_init_pos[idx_T], self.target_init_cov[idx_T])
                else:
                    self.target_est[i][idx_T].reset(
                        self.target_est_init_pos[idx_T,0:2], self.target_init_cov[idx_T,0:2,0:2])
                    

        # Used for episodic record index
        # self.i_episode = self.i_episode + 1
        
        
        # Append a new record unit when update
        self.robot_cov_trace_list.append(self.record_unit)
        self.target_cov_trace_list.append(self.record_unit)
        self.obstacles_count.append(self.obstacles_count_unit)
        # self.robot_obs_num_list = [[] for i in range(self.num_agent)]

        # self.robot_est[0].cov  = self.target_est[0].cov

        # set state
        
        Com_obs = np.identity(self.num_agent)
        self.state_func(Com_obs)
        return self.state

    def build_models(self):
        # Build a robot_true
        self.robot_true = [Agent(
            dim=3, sampling_period=self.sampling_period) for i in range(self.num_agent)]
        self.robot_est = [Agent_est(dim=3, sampling_period=self.sampling_period)
                          for i in range(self.num_agent)]

        self.target_true = [ Agent(dim=3, sampling_period=self.sampling_period) for j in range(self.num_target)]
        self.target_est = [[Agent_est(dim=3,sampling_period=self.sampling_period)\
                           for j in range(self.num_target)]for i in range(self.num_agent)]
        



    def step(self, action):
        # Record trace episodic trace
        for i in range(self.num_agent):
            self.robot_cov_trace_list[-1][i].append(
                trace(self.robot_est[i].cov))
            # self.robot_cov_trace_list[self.i_episode][i].append(
            #     self.robot_est[i].cov[2,2])
            # for j in range(self.num_target):
            self.target_cov_trace_list[-1][i].append(
                trace( self.target_est[i][0].cov ))
            # self.target_cov_trace_list[self.i_episode][i].append(
            #     self.target_est[i].cov[2,2])
            # if self.target_est[i][j].cov[2,2] > 0.11:
            #     check = 1
        
        info = []
        
        
        for i in range(self.num_agent):
            # Estimated robot pose propagation (t -> t+1)
            # vR_desired = action[i, 0]
            vR_desired = action[i][0]
            # omegaR_desired = action[i, 1]
            omegaR_desired = action[i][1]

            if self.parser.process_noise_fixed:
                self.sigma_vR = self.parser.sigma_vR
                self.sigma_wR = self.parser.sigma_wR
                
            else:
                self.sigmaR = self.sigmapR*vR_desired
                self.sigma_vR = self.sigmaR/sqrt(2)
                self.sigma_wR = 2*sqrt(2)*self.sigmaR

            desired_robot_action = np.array([vR_desired, omegaR_desired])
            self.robot_est[i].propagate(
                desired_robot_action, self.sigma_vR, self.sigma_wR)

            # True Robot pose Propagate
            vR_true = np.random.normal(vR_desired, self.sigma_vR)
            omegaR_true = np.random.normal(omegaR_desired, self.sigma_wR)

            true_robot_action = np.array([vR_true, omegaR_true])
            self.robot_true[i].propagate(true_robot_action)
            # TODO: the information
            info.append({})
        # Target propagation (t -> t+1)
        # Estimated target pose propagate



        

        # vT_desired, omegaT_desired = 
        # del_vx, del_vy = self.obstacle_detour_maneuver(

        # vT_desired,omegaT_desired = self.get_target_action()
        # TODO: design the disired trajectory
        # vT_desired = 0.25
        # omegaT_desired = 0.15

        [vT_desired, omegaT_desired] = self.target_controller(self.target_true[0].state)
        
        
        # Store for as state
        self.target_angular_velocity = omegaT_desired
        true_target_action = np.array([vT_desired, omegaT_desired])
        self.target_true[0].propagate(true_target_action)

        # estimated target pose propagate
        vT_est = np.random.normal(vT_desired, self.sigma_vT)
        omegaT_est = np.random.normal(omegaT_desired, self.sigma_wT)
        estimated_target_action = np.array([vT_est, omegaT_est])

        # desired_target_action = np.array([vT_desired, omegaT_desired])

        for i in range(self.num_agent):
            # for idx_T in range(self.num_target):
            self.target_est[i][0].propagate(
                estimated_target_action, self.sigma_vT, self.sigma_wT)


        # update
        if self.useupdate:
            zr, Rrb, Rb, Com_obs, RT_obs = self.measurement_generation()
            # observed = self.update(zr, Rrb, Rb, Com_obs, RT_obs)
            self.update(zr, Rrb, Rb, Com_obs, RT_obs)
            # self.observe_list.append(observed)
            self.com_plot = Com_obs
            self.RT_obs = RT_obs

        check = False
        # Debug: Check the cov goes to inf or not
        for i in range(self.num_agent):
            Pest = self.robot_est[i].cov
            array_sum = np.sum(Pest)
            array_has_nan = np.isnan(array_sum)
            if array_has_nan:
                print('array_has_nan3')
                check = 1

        # Check if we should terminate due to numerical issues
        if check:
            print("Warning: Too many numerical issues detected, terminating episode")
            done = [True] * self.num_agent

        # else:
            # self.observe_list.append(0)
        # Compute a reward from b_t+1|t+1 or b_t+1|t.
        if not self.useupdate:
            RT_obs=np.zeros((self.num_agent,self.num_agent+self.num_target))
        reward, done = self.get_reward(RT_obs)

        
        
        '''
        ## Judge if lose communication for 5 timesetp, if true fail 
        if Com_obs[0,1] == 0 :
            self.end_count = self.end_count+1
        else:
            self.end_count = 0
        if  self.end_count == 15:
            done = True
            # print('lose communication for a long time!')
        '''
        # update the true state
        self.state_func(Com_obs)
        
        self.truncated_indice += 1
        
        if self.truncated_indice > self.maxmum_run:
            self.truncated = True
        
        done = [d or self.truncated for d in done]  
            
        
        
        return [self.state, reward, done, self.truncated, info]

    def get_reward(self,RT_obs):
        ''''
        the reward function
        '''
        # initialize done with all false
        done = []
        for _ in range(self.num_agent):
            done.append(False)
            
        # inv_trace reward

        # reward = [(1/(1e-2+trace(self.target_est[i].cov)))
        #           for i in range(self.num_agent)]
        
        # reward = [1e5*( -1e-7*( trace(self.target_est[i][0].cov) -5)**3 )
        #           for i in range(self.num_agent)]
        
        # Reward Term 1: estimation to the target
        # Reward Term 2: estimation to robots state of itself
        reward = [( -10*( trace(self.target_est[i][0].cov))-5*( trace(self.robot_est[i].cov)))
                  for i in range(self.num_agent)]
        
    

        # reward = [( -1e-7*( trace(self.target_est[i][0].cov) )**3 )
        #           for i in range(self.num_agent)]
        # reward = [( 0) **3 
        #           for i in range(self.num_agent)]
        
        
        # TODO: remaining reward terms
        for i in range(self.num_agent):
            reward_obs = 0
            
            if RT_obs[i,self.num_agent ] == 1:
                self.observed_count = self.observed_count + 1
            else:
                self.unobserved_count = self.unobserved_count + 1
            
            if self.out_of_map(self.robot_est[i].state):
                reward_obs = 20
                self.obstacles_count[-1][i] = \
                            self.obstacles_count[-1][i] + 1 
                
            else:
                # Find the closest obstacle coordinate.
                obstacles_pt = self.MAP.get_closest_obstacle(self.robot_est[i].state,
                fov=2*np.pi,r_max=3)
                if obstacles_pt is None:
                    obstacles_pt = (self.max_range, np.pi)
                else:
                    if obstacles_pt[0] < 0.2:
                        reward_obs = 1.3
                        self.obstacles_count[-1][i] = \
                            self.obstacles_count[-1][i] + 1    
                            
                        # Collision with obstacles, the done will be true        
                        done[i] = True

            r, _ = cartesian2polar(self.target_est[i][0].state[0] - self.robot_est[i].state[0],
                                            self.target_est[i][0].state[1]-self.robot_est[i].state[1])
            
            reward_mutual_scale = 0.5
            reward_mutual = 0
            for j in range(self.num_agent):
                if i ==j:
                    continue
                if self.mutual_collision(self.robot_est[i].state,self.robot_est[j].state):
                    reward_mutual = reward_mutual_scale

                    self.obstacles_count[-1][i] = \
                        self.obstacles_count[-1][i] + 1   
                    break

            reward[i] = reward[i]  - reward_obs - reward_mutual
            
            



        return reward, done

    def  dis_obj(self,r,max_range,min_range):
        t = 0.1
        a = 2
        if r < min_range:
            obj = 0
        
        elif r < min_range+ a:
            u = min_range - r
            obj =  -np.log(-u/a)
            obj = 0
        elif r < max_range-a:
            obj = 0
        else:
            obj = (r-(max_range-a))**2
        return obj


    def state_func(self,Com_obs):
        """ self.state 
        Obtain the state based on the current state
        """
        # remark: used for update the true state
        self.state = zeros((self.num_agent, self.state_num ))
        state_old = self.state.copy() 

        # Calculate relative target state as state
        for i in range(self.num_agent):
            C = rotation_matrix(self.robot_est[i].state[2])
            
            i_p_T = C.T @ (self.target_est[i][0].state[0:2] -
                        self.robot_est[i].state[0:2]).reshape(2,)
            i_theta_T = self.target_est[i][0].state[2]-self.robot_est[i].state[2]

            self.state[i, 0:2] = ([i_p_T[0], i_p_T[1]])

            # Target relative orientation
            self.state[i, 2] = i_theta_T

            # Target Velocity Part
            self.state[i, 3:5] = (
                [self.target_linear_velocity, self.target_angular_velocity])

            # Target covariance Part
            self.state[i, 5] = trace(self.target_est[i][0].cov)

            '''
            idx = 0 
            for j in range(1,self.num_target):
                i_p_T_L = C.T @ (self.target_est[i][j].state[0:2] -
                self.robot_est[i].state[0:2]).reshape(2,)
                i_theta_T = self.target_est[i][j].state[2]- self.robot_est[i].state[2]
                
                # self.state[i, 6+idx*3:6+idx*3+3] = (
                #     [i_p_T_L[0], i_p_T_L[1],i_theta_T ])  
                self.state[i, 6+idx*3:6+idx*3+3] = (
                    [0,0,0])
                idx = idx + 1 
            '''

            id_b = 6
            # id_b = 6+3*(self.num_target-1)
            idx = 0
            for ell in range(self.num_agent):
                if ell == i:
                    continue
                else:      
                    id_n = ell 
                    # if Com_obs[i,ell] == 1:
                    i_p_R = C.T @ (self.robot_est[id_n].state[0:2] -
                                self.robot_est[i].state[0:2]).reshape(2,)
                    i_theta_R = self.robot_est[id_n].state[2]-self.robot_est[i].state[2]
                    self.state[i, id_b+idx*4:id_b+idx*4+3] = ([i_p_R[0], i_p_R[1], i_theta_R ])
                    self.state[i, id_b+idx*4+3] = trace(self.robot_est[id_n].cov)

                    # else:    
                        # NO update for the neighbor's state
                            
                        # i_p_R = C.T @ (self.robot_est_init_pos[id_n, 0:2] -
                        #             self.robot_est[i].state[0:2]).reshape(2,)
                        # i_theta_R = self.robot_est_init_pos[id_n,2]-self.robot_est[i].state[2]
                         
                    idx = idx + 1    

            # c_idx = id_b+idx*4+4



            if self.robot_est[i].state[0]<1 or self.robot_est[i].state[0]>35 or \
                self.robot_est[i].state[1]<1 or self.robot_est[i].state[1]>35:
                obstacles_pt = (self.max_range, np.pi)
            else:
                # Find the closest obstacle coordinate.
                obstacles_pt = self.MAP.get_closest_obstacle(self.robot_est[i].state)
                if obstacles_pt is None:
                    obstacles_pt = (self.max_range, np.pi)
                self.state[i, -6:-4] = ([obstacles_pt[0], obstacles_pt[1]]) 



            # Robot self Part
            self.state[i, -4:-1] = self.robot_est[i].state
            self.state[i, -1] = trace(self.robot_est[i].cov)

            # diff = state_old[i, 6:-6] - self.state[i, 6:-6] 
            # print(self.state[i, -4:-1])
            # print(self.state_old[i, -4:-1])
            # print( diff )
            
        # print()
        self.state = np.array(self.state).astype('float64')

    def measurement_generation(self):
        # Note that only first curr_meas_num nonzeros are actual measuremetns
        # Rrb = [np.zeros((0, 0)) for i in range(self.num_agent)]
        # Rb = [np.zeros((0, 0)) for i in range(self.num_agent)]
        nR = self.nR
        nT = self.nT

        Rrb = [[[] for i in range(self.nR+nT)] for i in range(self.nR)]
        Rb = [[[] for i in range(self.nR+nT)] for i in range(self.nR)]

        # R_obs = np.zeros((nR, nR), dtype=int)
        # T_obs = np.zeros((nR, nT), dtype=int)
        RT_obs = np.zeros((nR, nR+nT), dtype=int)
        # zr = np.zeros((nR, 3, nR+nT-1))  # robot_true-measurements
        zr = [[[] for i in range(self.nR+nT)] for i in range(self.nR)]

        Com_obs = np.eye(nR)
        # Com_obs = np.zeros((nR,nR))

        for ell in range(nR):
            total_meas_num = -1
            curr_meas_num = -1

            # Robot-Robot/Target measurements
            for j in range(nR+nT):
                # Just for test
                # if j < nR:
                #     continue

                if ell == j:
                    continue

                # if self.robot_est[ell].state[0]<1 or self.robot_est[ell].state[0]>35 or \
                #     self.robot_est[ell].state[1]<1 or self.robot_est[ell].state[1]>35:
                #     r, th = cartesian2polar(self.robot_true[j].state[0] - self.robot_true[ell].state[0],
                #                                 self.robot_true[j].state[1]-self.robot_true[ell].state[1])

                #     block  = True
                # else:            

                if j < nR:
                    r, th = cartesian2polar(self.robot_true[j].state[0] - self.robot_true[ell].state[0],
                                            self.robot_true[j].state[1]-self.robot_true[ell].state[1])

                    if self.robot_est[ell].state[0]<self.MAP.mapmin[0] or self.robot_est[ell].state[0]>self.MAP.mapmax[0] or \
                        self.robot_est[ell].state[1]<self.MAP.mapmin[1]  or self.robot_est[ell].state[1]>self.MAP.mapmax[1]:
                        block  = True
                    else:
                        block  = self.MAP.is_blocked( self.robot_true[ell].state,self.robot_true[j].state)                        
                
                else:
                    idx_T = j-nR
                    r, th = cartesian2polar(self.target_true[idx_T].state[0] - self.robot_true[ell].state[0],
                                            self.target_true[idx_T].state[1]-self.robot_true[ell].state[1])
                    

                    if self.robot_est[ell].state[0]<self.MAP.mapmin[0] or self.robot_est[ell].state[0]>self.MAP.mapmax[0] or \
                        self.robot_est[ell].state[1]<self.MAP.mapmin[1]  or self.robot_est[ell].state[1]>self.MAP.mapmax[1]:                        
                        block  = True

                    else:
                        # if idx_T ==0 :
                        block = self.MAP.is_blocked( self.robot_true[ell].state, self.target_true[idx_T].state) 
                        # print('target block',block)
                        # else:
                        #     block = False
                        if block == False:
                            check =1

                # measurement wrt robot_true
                th = pi_to_pi(th-self.robot_true[ell].state[2])

                if block == False:
                    check = 1


                if self.SIGPERCENT == 1:
                    sigma_r = self.sigma_p*r  # sigma_p percentage of range
                else:
                    sigma_r = self.sigma_r

                if r < self.max_com_range and j < nR:
                    # Com_obs[ell, j] = 0
                    Com_obs[ell, j] = 1

                # use measurement only if target is closer than max_range
                observed = r < self.max_range and r > self.min_range \
                            and abs(th) <= self.fov/2*(pi/180) \
                            and not(block)   
             
                if observed:
                    # T_obs[ell,j] = 1
                    RT_obs[ell, j] = 1
                    curr_meas_num = curr_meas_num+1
                    # Rb[ell][j] = block_diag(Rb[ell], self.sigma_th**2)
                    Rb[ell][j] = self.sigma_th**2

                    # distance-bearing meausement
                    r = np.random.normal(r, sigma_r)
                    th = np.random.normal(th, self.sigma_th)
                    Rii = block_diag(sigma_r**2, self.sigma_th**2)
                    # Rrb[ell][j] = block_diag(Rrb[ell], Rii)
                    Rrb[ell][j] =  Rii

                    zr[ell][j] = [r, th]
                    total_meas_num = total_meas_num+1

        return zr, Rrb, Rb, Com_obs, RT_obs

    def update(self, zr_a, Rrb_a, Rb_a, Com_obs_a, RT_obs):
        global s_temp

        # robot_est_temp = self.robot_est

        # Robot Localization
        for det_id in range(self.num_agent):
            n_meas = len(Rb_a[det_id])
            # zr = np.squeeze(zr_a[det_id, :, :])
            # zr = zr_a[det_id]
            # deted_ids = zr[2, 0:n_meas].astype(np.int32).T

            # if len(deted_ids) > 0:
            if n_meas > 0:
                # if n_meas > 1000:
                i = -1
                S = zeros((n_meas+1, 3, 3))
                Y = zeros((3, n_meas+1))

                for deted_id in range(self.num_agent+self.num_target):
                    # just for test
                    # if deted_id < self.num_agent:
                    #     continue

                    if not len(zr_a[det_id][deted_id]):
                        continue
                    i += 1
                    # if RT_obs[det_id, self.nR] == 1:
                    z_real = zr_a[det_id][deted_id]
                    # self.observe_list[timestep] = 1
                    if deted_id < self.num_agent:
                        zhat, dH1, dH2 = self.measurement_model(
                            self.robot_est[det_id].state, self.robot_est[deted_id].state)
                    else:
                        deted_id_T = deted_id - self.num_agent
                        zhat, dH1, dH2 = self.measurement_model(
                            self.robot_est[det_id].state, self.target_est[det_id][deted_id_T].state)
                        if deted_id_T!=0:
                            dH2 = dH2[:,0:2]

                    rij = np.array(
                        [[z_real[0]-zhat[0]], [pi_to_pi(z_real[1]-zhat[1])]])
                    
                    # R = Rrb_a[det_id][i:i+2, i:i+2]
                    R = Rrb_a[det_id][deted_id]
                    # if R == []:
                    #     check = 1


                    # Self-Localization
                    if deted_id < self.num_agent:
                        dInvR_bar = pinv(
                            R + dH2 @ self.robot_est[deted_id].cov @ dH2.T)
                    else:
                        dInvR_bar = pinv(
                            R + dH2 @ self.target_est[det_id][deted_id_T].cov @ dH2.T)
                    # print('Robot    ',det_id)
                    # print('R=       ',trace(R))
                    # print('HSH=     ',trace(dH2 @ self.target_est[det_id].cov @ dH2.T))

                    # if trace(R) < trace(dH2 @ self.target_est[det_id].cov @ dH2.T):
                    #     check = 1

                    dZr_bar = rij + \
                        (dH1 @ self.robot_est[det_id].state).reshape(2, 1)
                    s = dH1.T @ dInvR_bar @ dH1
                    y = dH1.T @ dInvR_bar @ dZr_bar
                    # print('fused=   ',trace( pinv(s)[0:2,0:2] ))

                    if trace(pinv(s)) < 0:
                        check = 1

                    S[i, :, :] = s
                    # print()
                    Y[:, i] = y.reshape(3,)

                Omega = pinv(self.robot_est[det_id].cov)
                q = Omega@self.robot_est[det_id].state
                S[i+1, :, :] = Omega
                Y[:, i+1] = q
                # print('prior=   ',trace(pinv(Omega[0:2,0:2])))
                # s_temp = S

                # print('det_id=',det_id)
                Pest, Xest = self.CI(S, Y, print_weight=0, xyfusion=0,plot=0)
                if abs(Pest[0,1] - Pest[1,0]) > 1e-3:
                    check = 1
                # if det_id == 0:
                #     Pest, Xest = self.CI(S, Y, print_weight=0, xyfusion=0,plot=0)
                # else:
                #     Pest, Xest = self.CI(S, Y, print_weight=0, xyfusion=0)
                    # print()
                # if (trace(Pest) > trace(pinv(s))) and (trace(Pest) > trace(self.robot_est[det_id].cov)):
                #     check = 1
                
                self.robot_est[det_id].state = Xest
                self.robot_est[det_id].cov = Pest

                # Debug
                array_sum = np.sum(Pest)
                array_has_nan = np.isnan(array_sum)
                if array_has_nan:
                    chenck = 1
                    print('array_has_nan1')

            # end if n_meas > 0:

            # ---------------------------------------------------------------------
            # Target Tarcking (one target case)
            for ell_T in range(self.num_target):
                # ell = 1
                if ell_T == 4:
                    check =1
                if ell_T ==0:
                    state_dim = 3
                else:
                    state_dim = 2
                id_T = self.num_agent+ell_T
                det_R = np.where(np.squeeze(RT_obs[:, id_T]) == 1)[
                    0].tolist()
                # Com_obs[det_id-1] = 1
                Com_nei = np.where(Com_obs_a[det_id, :] == 1)[0].tolist()
                
                # communication neighbor that detect the target as well
                det_and_com_neighbor = list(set(det_R) & set(Com_nei))
                n_meas = len(det_and_com_neighbor)
                if(n_meas) > 0:
                    if ell_T !=0:
                        check = 1
                    i = -1
                    S = zeros((n_meas, state_dim, state_dim))
                    Y = zeros((state_dim, n_meas))

                    for det_robot in det_and_com_neighbor:
                        # det_all = len(np.where(np.squeeze(RT_obs[det_robot,:]) == 1)[0].tolist())-1
                        i += 1
                        # id_T = len(np.where(RT_obs[det_robot, :] == 1)[0].tolist())-1
                        # id_T = np.where(zr_a[det_id, 2,: ] == self.num_agent)

                        s,y = self.generate_information_pair(zr_a,Rrb_a,det_robot,ell_T,state_dim)

                        S[i, :, :] = s
                        Y[:, i] = y

                    # s_temp = S
                    S_est, Y_est = self.CI(S, Y, infomation_form=1)

                    n_coms = len(Com_nei)
                    if ell_T == 0:
                        Omega = zeros((n_coms, 3, 3))
                        Q = zeros((3, n_coms))
                    else:
                        Omega = zeros((n_coms, 2, 2))
                        Q = zeros((2, n_coms))
                    
                    i = -1
                    # prior
                    for com_nei in Com_nei:
                        i += 1
                        omega = pinv(self.target_est[com_nei][ell_T].cov)
                        q = omega@self.target_est[com_nei][ell_T].state
                        Omega[i, :, :] = omega
                        if ell_T == 0:
                            Q[:, i] = q.reshape(3,)
                        else:
                            Q[:, i] = q.reshape(2,)

                    # s_temp = Omega
                    S_prior, Y_prior = self.CI(Omega, Q, infomation_form=1)

                    # fused estimate
                    # S_fused =
                    if ell_T !=0:
                        S_fused = zeros((2, 2, 2))
                        Y_fused = zeros((2, 2))

                        S_fused[0, :, :] = S_est[0:2,0:2]
                        S_fused[1, :, :] = S_prior[0:2,0:2]

                        Y_fused[:, 0] = Y_est[0:2]
                        Y_fused[:, 1] = Y_prior[0:2]
                    else:
                        S_fused = zeros((2, 3, 3))
                        Y_fused = zeros((3, 2))

                        S_fused[0, :, :] = S_est
                        S_fused[1, :, :] = S_prior

                        Y_fused[:, 0] = Y_est
                        Y_fused[:, 1] = Y_prior

                    # s_temp = S_fused
                    if ell_T == 0:
                        Pest, Xest = self.CI(S_fused, Y_fused,no_weight=0, print_weight=0)
                    else:
                        Pest, Xest = self.CI(S_fused, Y_fused,no_weight=0, print_weight=0)

                    self.target_est[det_id][ell_T].state = Xest
                    self.target_est[det_id][ell_T].cov = Pest

                    # if self.target_est[det_id].cov[2,2] > 0.11:
                    #     check = 1
                    # print('target prior trace =     ' +
                    #       str(trace(pinv(S[1, :, :]))))
                    # print('target measure trace =   ' +
                    #       str(trace(pinv(S[0, :, :]))))
                    # # print('reward =             ' + str(reward))
                    # print('target fused trace =     ' + str(trace(Pest)))
                    # print()

                    # observed
                    # return 1
                elif Com_nei != []:
                    n_coms = len(Com_nei)
                    #state_dim = 3 if ell_T = 0 else 2
                    Omega = zeros((n_coms, state_dim, state_dim))
                    Q = zeros((state_dim, n_coms))
                    i = -1
                    # prior
                    for com_nei in Com_nei:
                        i += 1
                        omega = pinv(self.target_est[com_nei][ell_T].cov)
                        q = omega@self.target_est[com_nei][ell_T].state
                        Omega[i, :, :] = omega
                        if ell_T == 0:
                            Q[:, i] = q.reshape(3,)
                        else:
                            Q[:, i] = q.reshape(2,)

                    # s_temp = Omega
                    Pest, Xest = self.CI(Omega, Q)

                    self.target_est[det_id][ell_T].state = Xest
                    self.target_est[det_id][ell_T].cov = Pest

                # observed
                # return 0
        # end for det_id in range(self.num_agent)
        # self.robot_est = robot_est_temp

    def measurement_model(self, xe_i, xe_j):
        # xe_i = det_state
        # xe_j = deted_state
        # range and bearing meausrement model

        J = np.array([[0, -1],
                      [1, 0]])

        # xe_i = self.robot_est.state
        # xe_j = self.target_est.state
        C = rotation_matrix(xe_i[2])
        Pij = C.T @ (xe_j[0:2] - xe_i[0:2])

        rho = norm(Pij)
        th = np.arctan2(Pij[1], Pij[0])
        zhat = np.array([rho, th])
        # %jacobians evaluated at k+1/k
        # H_Lk = np.array([1 / norm(Pij) * Pij.cT, OMPCSEMI, 1 / norm(Pij) ** 2 * Pij.cT * J.cT])

        H_Lk = np.zeros((2, 2))
        H_Lk[0, 0:2] = np.array([1/norm(Pij)*Pij.T])
        H_Lk[1, 0:2] = np.array([1/norm(Pij)**2 * Pij.T.dot(J.T)])

        # Hi_prior =

        Hi_prior = np.block([
            [eye(2),  J@(xe_j[0:2]-xe_i[0:2]).reshape(2, 1)]
        ])

        Hi = -H_Lk @ C.T @ Hi_prior
        Hj = np.zeros((2, 3))
        # Hj[0:2,0:2] = H_Lk @ C.T

        Hj = np.block([
            [H_Lk @ C.T,  zeros((2, 1))]
        ])

        return zhat, Hi, Hj

    # def get_list(self):
    #     return self.robot_cov_trace_list, self.target_cov_trace_list

    def plot_trace(self):
        # plt.figure(1)
        t = np.arange(0.0, 50.0, 0.5)
        # plt.plot(t,)    # plt.plot(t,)

        average_episodic_cov_trace = [list(map(mean, layer)) for layer in self.robot_cov_trace_list]

        ax1 = plt.subplot(311)
        for i in range(self.num_agent):
            plt.plot(t, self.robot_cov_trace_list[i], label='Robot'+str(i))
        plt.setp(ax1.get_xticklabels())
        ax1.set_title('robot cov')
        ax1.legend()

        # share x only
        ax2 = plt.subplot(312)
        for i in range(self.num_agent):
            for j in range(self.num_target):
                plt.plot(
                    t, self.target_cov_trace_list[i][j], label='Robot'+str(i))
        # make these tick labels invisible
        plt.setp(ax2.get_xticklabels())
        ax2.set_title('target cov')
        ax2.legend()

        # RMSE calculation
        for i in range(self.num_agent):
            self.robot_rmse[i] = self.robot_est_traj[i] - self.robot_traj[i]

        plt.show()

    def generate_information_pair(self,zr_a,Rrb_a,det_robot,ell_T,state_dim):
        id_T = self.num_agent+ell_T
        z_real = zr_a[det_robot][id_T]

        zhat, dH1, dH2 = self.measurement_model(
            self.robot_est[det_robot].state, self.target_est[det_robot][ell_T].state)

        rij = np.array(
            [[z_real[0]-zhat[0]], [pi_to_pi(z_real[1]-zhat[1])]])

        # R = Rrb_a[det_robot][2*id_T:2*id_T+2, 2*id_T:2*id_T+2]
        R = Rrb_a[det_robot][id_T]

        if ell_T != 0:
            # dH1 = dH1[:,0:2]
            dH2 = dH2[:,0:2]

        dInvR_bar = pinv(
            R + dH1 @ self.robot_est[det_robot].cov @ dH1.T)
        dZr_bar = rij + \
            (dH2 @ self.target_est[det_robot][ell_T].state).reshape(2, 1)
        s = dH2.T@dInvR_bar@dH2
        y = dH2.T@dInvR_bar@dZr_bar

        y = y.reshape(state_dim,)
        # if ell_T == 0:
        #     y.reshape(3,)
        # else:
        #     y.reshape(2,)
        return s,y


    def CI(self, s, y, infomation_form=0, no_weight=0, print_weight=0, xyfusion=0,plot=0):

        # optimize
        S_fused = zeros((s.shape[1], s.shape[1]))
        x_fuesd = zeros((y.shape[0], 1))

        s_store = s
        if xyfusion:
            s = s[:, 0:2, 0:2]
        global s_temp
        # s_temp = s
        s_temp = s

        N = s.shape[0]
        if N > 2:
            check = 1

        if no_weight:
            c = ones((N))/N
            c[0] = 1.0
            c[1] = 0.0

        else:
            x0 = ones((N))/N
            # print(x0)

            # x0 = zeros((N))
            # x0[0] = 0.0
            # x0[1] = 1.0

            # for the range of boundaries: consistant with the mentioned cases
            b = (0.0, 1.0-1e-10)
            # bnds = b
            bnds = tuple([b for i in range(N)])
            # self.robot_cov_trace_list = [[] for i in range(self.num_agent)]

            # print('Original Objective: ' + str(objective(x0)))
            con2 = {'type': 'eq', 'fun': constraint}
            cons = ([con2])
            try:
                solution = minimize(objective, x0, method='SLSQP',  # SLSQP
                                bounds=bnds, constraints=cons,
                                options={'ftol': 1e-10})
            except:
                solution = minimize(objective, x0, method='SLSQP',  # SLSQP
                                bounds=bnds, constraints=cons,
                                options={'ftol': 1e-10})

            c = solution.x
            # print(c)
            if (c[0] == 0.00) and (not infomation_form):
                check = 1
                # print(check)
            if plot:
                CI_plot(s,y)
           
        s = s_store
        # c = x
        # calculate fuised covariance and state
        for i in range(s.shape[0]):
            S_fused = S_fused + c[i]*s[i, :, :]

        if infomation_form:
            return S_fused, y@c

        else:
            P_fused = pinv(S_fused)
            x_fuesd = P_fused@(y@c)

            if print_weight:
                # if c[0] > c[1]:
                print('c= ',end=' ')
                for c_i in c:
                    print('{:.3f}'.format(c_i) ,end=' ')
                print()

                # w1,_ = eig(s[0, :, :])
                # w2,_ = eig(s[1, :, :])      

                # print('eig_mea = ', w1 )
                # print('eig_pri = ', w2 )

                # print('trace = ', trace(pinv(s[0, :, :])), 
                #         trace( pinv(s[1, :, :]) ),trace( P_fused ) )
                print()


            return P_fused, x_fuesd

    def get_target_action(self):
        self.turn_time = 4
        distance_interval = [7,6,5,5,6,5,18,24,18,12,6,5]
        time_interval = np.array(distance_interval)*4
        # 1
        if self.time_step_count > sum(time_interval[:1]) and \
            self.time_step_count <= sum(time_interval[:1])+1*self.turn_time:

            vT_desired = 0 
            omegaT_desired = -pi/4
            
        # 2
        elif self.time_step_count > sum(time_interval[:2])+1*self.turn_time and \
            self.time_step_count <= sum(time_interval[:2])+2*self.turn_time:

            vT_desired = 0 
            omegaT_desired = -pi/4

        # 3
        elif self.time_step_count > sum(time_interval[:3])+2*self.turn_time and \
            self.time_step_count <= sum(time_interval[:3])+3*self.turn_time:
            vT_desired = 0 
            omegaT_desired = pi/4
        # 4
        elif self.time_step_count > sum(time_interval[:4])+3*self.turn_time and \
            self.time_step_count <= sum(time_interval[:4])+4*self.turn_time:
            vT_desired = 0 
            omegaT_desired = -pi/4
        
        # 5
        elif self.time_step_count > sum(time_interval[:5])+4*self.turn_time and \
            self.time_step_count <= sum(time_interval[:5])+5*self.turn_time:
            vT_desired = 0 
            omegaT_desired = -pi/4
        # 6
        elif self.time_step_count > sum(time_interval[:6])+5*self.turn_time and \
            self.time_step_count <= sum(time_interval[:6])+6*self.turn_time:
            vT_desired = 0 
            omegaT_desired = pi/4
        # 7
        elif self.time_step_count > sum(time_interval[:7])+6*self.turn_time and \
            self.time_step_count <= sum(time_interval[:7])+7*self.turn_time:
            vT_desired = 0 
            omegaT_desired = pi/4
        # 8
        elif self.time_step_count > sum(time_interval[:8])+7*self.turn_time and \
            self.time_step_count <= sum(time_interval[:8])+8*self.turn_time:
            vT_desired = 0 
            omegaT_desired = pi/4
        # 9
        elif self.time_step_count > sum(time_interval[:9])+8*self.turn_time and \
            self.time_step_count <= sum(time_interval[:9])+9*self.turn_time:
            vT_desired = 0 
            omegaT_desired = pi/4
        # 10
        elif self.time_step_count > sum(time_interval[:10])+9*self.turn_time and \
            self.time_step_count <= sum(time_interval[:10])+10*self.turn_time:
            vT_desired = 0 
            omegaT_desired = pi/4
        # 11
        elif self.time_step_count > sum(time_interval[:11])+10*self.turn_time and \
            self.time_step_count <= sum(time_interval[:11])+11*self.turn_time:
            vT_desired = 0 
            omegaT_desired = pi/4
        # 11
        elif self.time_step_count > sum(time_interval[:12])+11*self.turn_time:
            vT_desired = 0 
            omegaT_desired = 0
        else:
            vT_desired = 0.5
            omegaT_desired = 0.0

        return vT_desired,omegaT_desired

    def out_of_map(self,state):
        if state[0]<(self.MAP.mapmin[0]+0.1) or (state[0]>self.MAP.mapmax[0]-0.1) or \
            state[1]<(self.MAP.mapmin[1]+0.1)  or (state[1]>self.MAP.mapmax[1]-0.1) :
            return True
        else:
            return False
    def mutual_collision(self,ri_state,rj_state,safe_distance=0.4):
        r,_ = cartesian2polar(ri_state[0] - rj_state[0],
                                            ri_state[1]-rj_state[1])
        if r < safe_distance:
            return True
        else:
            return False


def objective(c):
    global s_temp
    S_fused = zeros(s_temp.shape[1:3])
    for i in range(s_temp.shape[0]):
        S_fused = S_fused + c[i]*s_temp[i, :, :]
    # print('S_fused= ',S_fused)
    P_fused = pinv(S_fused)
    res = trace(P_fused)
    # rank_fuse = matrix_rank(S_fused)
    return res

def objective_test(c):
    global s_temp
    S_fused = zeros((3, 3))
    for i in range(s_temp.shape[0]):
        S_fused = S_fused + c[i]*s_temp[i, :, :]
    # print('S_fused= ',S_fused)
    # print('P_fused= ',pinv(S_fused))
    value =  trace(S_fused)
    P_fused = pinv(S_fused)
    rank_fuse = matrix_rank(S_fused)
    value = trace(P_fused)
    return value


def constraint(c):
    # global s_temp
    A = ones((c.shape[0]))
    sum_eq = A@c-1
    return sum_eq


def CI_plot(s, y):
    fig = plt.figure(3)
    fig.clf()

    c0 = np.linspace(0, 1, 1000)
    c1 = zeros((c0.shape[0]))

    tr_p_fuse = zeros((c0.shape[0]))
    rank_p_fuse  = zeros((c0.shape[0]))
    # c1 = 1-c0
    # y = x**2
    for i in range(c0.shape[0]):
        c1[i] = 1-c0[i]
        c = np.array([c0[i], c1[i]])
        tr_p_fuse[i] = objective(c)
        rank_p_fuse[i] = objective_test(c)

    # ax1 = fig.subplots(121)
    # plt.plot(c0, tr_p_fuse)

    ax2 = fig.subplots()
    plt.plot(c0, rank_p_fuse)
    # plt.show()
    plt.draw()
    plt.pause(0.0001)
    # plt.pause(5)
    check = 1
