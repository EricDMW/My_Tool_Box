#!opt/anaconda3/envs/lie_optimal/bin python
# _*_ coding: utf-8 _*_
'''
@File  : llm_para.py
@Author: Dongming Wang
@Email : dongming.wang@email.ucr.edu
@Date  : 10/12/2024
@Time  : 20:55:34
@Info  : TBC
'''

import argparse


def get_llm_config():
    """
    Function: get_llm_para
    Description: TBC
    """

    parser = argparse.ArgumentParser(description="llm mode parameters")
    parser.add_argument('--model_name', type=str, default="gpt-4o")
    parser.add_argument('--cen_decen_framework', type=str, default="DMAS")
    parser.add_argument('--linear_speed', type=float, default=1.0)
    parser.add_argument('--dialogue_history_method', type=str, default='_w_only_state_action_history')
    parser.add_argument('--save_path', type=str, default='llm/data')
    return parser