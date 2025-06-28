#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : target_control_para.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: RL_LLM
@Date   : 10/16/2024
@Time   : 16:33:49
@Info   : Parameters used for target trajectory plan
"""

import argparse

def get_target_control_config():
    """
    Description of the function get_target_control_config.
    """
    parser = argparse.ArgumentParser(description="llm mode parameters")
    parser.add_argument('--k_p', type=float, default=1, help='Proportional gain for x[1] position control')
    parser.add_argument('--k_theta', type=float,default=0.5,help='Proportional gain for heading control ')
    return parser