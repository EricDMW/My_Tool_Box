#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : target_moving.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: RL_LLM
@Date   : 10/16/2024
@Time   : 14:57:54
@Info   : Description of the script
"""
import torch
import warnings
import numpy as np
from scipy.interpolate import CubicSpline

class Target_Moving:
    '''
    Design the moving of the target along given trajectories
    '''
    
        
    @classmethod
    def Choose_map(cls, parser):
        """
        Based on map, choose different moving policies.
        """
        cls.target_velocity = parser.target_linear_velocity
        cls.target_omega_bound = parser.target_omega_bound
        cls.k_p = parser.k_p
        cls.k_theta = parser.k_theta  # Proportional gain for heading control
        cls.map_name = parser.map_name
        
        # Empty map trajectory control parameter: initialization
        cls.circling = False
        cls.border = 32
        cls.magnitude = 5
        cls.freq = 1
        
        match cls.map_name:
            case "empty":
                cls.target_action = cls.empty
            case "obstacles05":
                cls.target_action = cls.obstacle_05
            case "obstacles04":
                cls.target_action = cls.obstacle_04
            case _:
                warnings.warn("Moving method does not specifically designed, \
                    continue with randome moving type in random,\
                    continue with circles type in circle",category=UserWarning)
                moving_policy = input("Plese enter moving policy")
                match moving_policy:
                    case "random":
                        cls.target_action = cls.randmoving
                    case "circle":
                        cls.target_action = cls.circlemoving
                    case _:
                        raise NotImplementedError
                            

        
    ###################### EMPTY MAP TAEGET TRAJECTORY#################################    
    @classmethod
    # Define the trajectory for empty map
    def empty_trajectory(cls,x1):
        # gradient of the trajectory
        return cls.magnitude * torch.cos(cls.freq*torch.tensor(x1))  # Change this to any trajectory you want
    
    @classmethod
    def empty(cls,target_status):
        """
        Description of the function empty.
        """
        # Alert: Not automatically tuned parameter 35, used for design trajectory
        if target_status[0] < cls.border and not cls.circling:
            # Desired trajectory: x^2 = sin(x^1)
            x1_desired = target_status[0]
            x2_desired = cls.empty_trajectory(x1_desired)

            

            # Control law for forward velocity v
            v = cls.target_velocity  # Nominal forward velocity
            
            # Desired heading to follow x^2 = sin(x^1)
            theta_desired = torch.atan(x2_desired)
            
            # Control law for angular velocity omega
            omega = cls.k_theta * (theta_desired - target_status[2])
            
            
        else:
            cls.circling = True
            radius = 4
            v = 0.25
            omega = -v / radius
        
        return torch.tensor([v, omega])
    
    
    ###################### Obstacle map 05 MAP TAEGET TRAJECTORY#################################    
    @classmethod
    def obstacle_05(cls, target_status):
        """
        Compute control command [v, omega] for moving the target along a smooth,
        obstacle-avoiding trajectory toward the upper-left corner on 'obstacles04' map.
        
        Parameters:
            target_status (list or tensor): [x, y, theta]
        
        Returns:
            torch.Tensor: [v, omega] velocity command
        """
        # === Constants ===
        v_nominal = 0.25
        epsilon = 1e-6  # for numerical stability

        # === Extract state ===
        x, y, theta = target_status

        # === Define trajectory waypoints ===
        waypoints = [
            ((-np.inf, 35.0), (35.0, 10.0)),
            ((35, 38),    (38, 40)),
            ((38, 40),  (40, 66.0)),
            ((40, np.inf), (69, 67))
        ]

        # === Find the active segment and compute control ===
        for (x_range, (x_t, y_t)) in waypoints:
            if x_range[0] <= x < x_range[1]:
                dx = x_t - x
                dy = y_t - y
                if abs(dx) < epsilon and abs(dy) < epsilon:
                    return torch.tensor([0.0, 0.0])

                theta_target = np.arctan2(dy, dx)
                theta_error = (theta_target - theta + np.pi) % (2 * np.pi) - np.pi
                omega = cls.k_theta * theta_error
                return torch.tensor([v_nominal, omega])

        # === If at final goal ===
        return torch.tensor([0.0, 0.0])
    
        
    ###################### Obstacle map 04 MAP TAEGET TRAJECTORY#################################    
    
    @classmethod
    def obstacle_04(cls, target_status,*kwargs):
        """
        Compute control command [v, omega] for moving the target along a smooth,
        obstacle-avoiding trajectory toward the upper-left corner on 'obstacles04' map.
        
        Parameters:
            target_status (list or tensor): [x, y, theta]
        
        Returns:
            torch.Tensor: [v, omega] velocity command
        """
        # === Constants ===
        v_nominal = 0.1
        epsilon = 1e-6  # for numerical stability

        # === Extract state ===
        x, y, theta = target_status

        # === Define trajectory waypoints ===
        waypoints = [
            ((-np.inf, 20), (20.0, 10.0)),
            ((20, 22.5),    (22.5, 15.5)),
            ((22.5, 32.5),  (32.5, 15.8)),
            ((32.5, np.inf), (33.5, 32.5))
        ]

        # === Find the active segment and compute control ===
        for (x_range, (x_t, y_t)) in waypoints:
            if x_range[0] <= x < x_range[1]:
                dx = x_t - x
                dy = y_t - y
                if abs(dx) < epsilon and abs(dy) < epsilon:
                    return torch.tensor([0.0, 0.0])

                theta_target = np.arctan2(dy, dx)
                theta_error = (theta_target - theta + np.pi) % (2 * np.pi) - np.pi
                omega = cls.k_theta * theta_error
                return torch.tensor([v_nominal, omega])

        # === If at final goal ===
        return torch.tensor([0.0, 0.0])
    
    
    
    ######################        Default Moving Startegies     ################################# 
    @classmethod
    def randmoving(cls,*kwargs):
        # Random Moving
        epsilon = 1e-6  # Small value to ensure it's in the range
        vT_desired = 0.25
        omegaT_desired = -torch.pi/4 + (torch.pi/2 - epsilon) * torch.rand(1) + epsilon / 2
        return vT_desired, float(omegaT_desired)
    @classmethod
    def circlemoving(cls,target_status, *kwargs):
        # Start from current location, start doing circling
        vT_desired = 0.25
        omegaT_desired = 0.15
        return vT_desired, omegaT_desired 