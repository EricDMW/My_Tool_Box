#!opt/anaconda3/envs/lie_optimal/bin python
# _*_ coding: utf-8 _*_
'''
@File  : ajlatt_para.py
@Author: Dongming Wang
@Email : dongming.wang@email.ucr.edu
@Date  : 10/09/2024
@Time  : 19:26:06
@Info  : Parameters used in ajlatt
'''

import argparse
import numpy as np
from numpy import pi

def get_ajlatt_config():
    """
    The configuration parser for the target tracking environment.
    """
    parser = argparse.ArgumentParser(description="Target Tracking Configurations")

    # General parameters
    parser.add_argument('--version', type=int, default=0, help='Version of the configuration')
    parser.add_argument('--episode_num', type=int, default=200, help='Number of episodes')
    
    # Sensor parameters
    parser.add_argument('--sensor_r_max', type=float, default=3.0, help='Maximum sensor range')
    parser.add_argument('--sensor_r_min', type=float, default=0.0, help='Minimum sensor range')
    parser.add_argument('--fov', type=float, default=180.0, help='Field of view in degrees')
    parser.add_argument('--sigma_p', type=float, default=0.0, help='Sensor range noise')
    parser.add_argument('--sigma_r', type=float, default=0.0, help='Sensor bearing noise')
    parser.add_argument('--sigma_th', type=float, default=0.0, help='Sensor bearing noise')
    parser.add_argument('--SIGPERCENT', type=int, default=1, help='Noise is proportional to distance')

    # Communication parameters
    parser.add_argument('--useupdate', type=int, default=1, help='Whether to use updates (1) or not (0)')
    parser.add_argument('--commu_r_max', type=float, default=6.0, help='Maximum communication range')

    # Robot parameters
    parser.add_argument('--num_Robot', type=int, default=4, help='Number of robots')
    
    parser.add_argument('--robot_est_init_pos', type=float, nargs=3,
                        default=[0.0, 0.0, 0.0],
                        help='Robot estimated initial position (x, y, theta)')
    parser.add_argument('--robot_init_cov', type=float, nargs='+',
                        default=(1e-1 * np.ones((6))).tolist(),
                        help='Robot initial covariance values for each robot'
                    )

    parser.add_argument('--process_noise_fixed', type=int, default=1, help='Process noise fixed (1) or not (0)')
    parser.add_argument('--sigmapR', type=float, default=0.1, help='Process noise for robot position')
    parser.add_argument('--sigma_vR', type=float, default=0.1, help='Process noise for robot linear velocity')
    parser.add_argument('--sigma_wR', type=float, default=0.5 / 180 * pi, help='Process noise for robot angular velocity')

    # Target parameters
    parser.add_argument('--num_Target', type=int, default=1, help='Number of targets')
    
    parser.add_argument('--target_init_cov', type=float, nargs='+',
                        default=[1.0] + [10.0] * 10,
                        help='Target initial covariance (list of values)')

    parser.add_argument('--target_linear_velocity', type=float, default=0.3, help='Target linear velocity')
    parser.add_argument('--target_omega_bound', type=float, default=pi / 4, help='Target angular velocity bound')
    parser.add_argument('--sigmapT', type=float, default=0.1, help='Process noise for target position')
    parser.add_argument('--sigma_vT', type=float, default=0.2, help='Process noise for target linear velocity')
    parser.add_argument('--sigma_wT', type=float, default=0.5 / 180 * pi, help='Process noise for target angular velocity')

    # Environment parameters
    parser.add_argument('--T_steps', type=int, default=120, help='Number of time steps')
    parser.add_argument('--sampling_period', type=float, default=0.5, help='Sampling period')
    parser.add_argument('--margin2wall', type=float, default=1.0, help='Margin distance to wall')
    
    
    # 
    parser.add_argument(
    '--env_name',
    type=str,
    default='TargetTracking-v0',
        help='environment ID'
    )
    parser.add_argument(
        '--render',
        type=bool,
        default=True,
        choices=[True, False],
        help='whether to render (1: Yes, 0: No)'
    )
    parser.add_argument(
        '--record',
        type=bool,
        default=False,
        help='whether to record (1: Yes, 0: No)'
    )
    parser.add_argument(
        '--ros',
        type=int,
        default=0,
        help='whether to use ROS (1: Yes, 0: No)'
    )
    parser.add_argument(
        '--num_targets',
        type=int,
        default=1,
        help='the number of targets to track'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='.',
        help='a path to a directory to log your data'
    )
    parser.add_argument(
        '--map_name',
        type=str,
        default='obstacles04',
        # default='obstacles05',
        choices=['empty','obstacles05','obstacles04'],
        help='Name of the map to use'
    )
    parser.add_argument(
        '--repeat',
        type=int,
        default=1,
        help='number of times to repeat the experiment'
    )
    parser.add_argument(
        '--im_size',
        type=int,
        default=28,
        help='size of the input images (width and height)'
    )
    


    # Position parameters
    parser.add_argument('--target_init_pose', type=float, nargs='+',
                        default=[
                            [10.5, 11.5, 0.0],
                            [13.5, 6.0, 0.0], 
                            [20.0, 10.0, 0.0], 
                            [10.0, 3.0, 0.0], 
                            [15.0, 23.0, 0.0], 
                            [25.0, 3.0, 0.0], 
                            [30.0, 20.0, 0.0],
                            [10.0, 15.0, 0.0], 
                            [20.0, 20.0, 0.0], 
                            [25.0, 10.0, 0.0], 
                            [30.0, 10.0, 0.0],
                            [10.0, 10.0, 0.0], 
                            [12.0, 30.0, 0.0]
                        ],
                        help='Target initial positions (x, y, theta) for each target'
                    )

    parser.add_argument('--robot_init_pose', type=float, nargs='+',
                        default=[
                            [7.5, 12.5, 0.0], [7.5, 10.5, 0.0], [10, 12.5, 0.0], [10.0, 10.5, 0.0],
                            [80.0, 10.0, -np.pi], [40.0, -20.0, np.pi/2]
                        ],
                        help='Robot initial positions (x, y, theta) for each robot'
                    )
    # truncated parameters
    parser.add_argument('--maxmum_run', type=int, default=1000,help='Truncated condition enven there is no any termination')
 
    parser.add_argument('--fix_seed', type=bool, default=False, help='fix the random seed for both numpy and torch or not')
    
    return parser
