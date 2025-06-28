#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : __init__.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: RL_LLM
@Date   : 10/06/2024
@Time   : 19:13:23
@Infor  : Initialize all parameters
"""
# # Solve Potential Path Issues
# import sys
# from pathlib import Path
# # Add parameters to path
# parent_dir = Path(__file__).resolve().parent
# sys.path.append(str(parent_dir))
import argparse
import numpy as np

from .ajlatt_para import get_ajlatt_config
from .debug_para import get_debug_config
from .llm_para import get_llm_config
from .target_control_para import get_target_control_config
from .ppo_config import get_config_ppo

def get_default_params():
    """
    Returns the default parameters as a dictionary from ajlatt parser.
    """
    parser = get_ajlatt_config()
    args, _ = parser.parse_known_args()
    
    # Convert args to dictionary
    params = vars(args)
    
    # Convert target_init_pose and robot_init_pose to numpy arrays
    if 'target_init_pose' in params:
        target_poses = params['target_init_pose']
        # Reshape the flat list into 2D array
        if len(target_poses) > 0:
            params['target_init_pose'] = np.array(target_poses).reshape(-1, 3)
    
    if 'robot_init_pose' in params:
        robot_poses = params['robot_init_pose']
        # Reshape the flat list into 2D array
        if len(robot_poses) > 0:
            params['robot_init_pose'] = np.array(robot_poses).reshape(-1, 3)
    
    # Convert robot_init_cov to numpy array
    if 'robot_init_cov' in params:
        params['robot_init_cov'] = np.array(params['robot_init_cov'])
    
    # Convert target_init_cov to numpy array
    if 'target_init_cov' in params:
        params['target_init_cov'] = np.array(params['target_init_cov'])
    
    return params

def combine_parsers(*parsers):
    """
    Combine arguments from multiple parsers into one.
    
    Arguments:
    *parsers -- a list of parser objects to combine
    
    Returns:
    Combined ArgumentParser object.
    """
    # Create a new parser
    combined_parser = argparse.ArgumentParser(description="Combined parser")

    # Loop through the provided parsers and add their arguments to the new parser
    for parser in parsers:
        for action in parser._actions:
            # Avoid adding default 'help' actions multiple times
            if not isinstance(action, argparse._HelpAction):
                combined_parser._add_action(action)
    
    return combined_parser


all_parsers = combine_parsers(get_debug_config(), get_ajlatt_config(),get_llm_config(),get_target_control_config(),get_config_ppo())
