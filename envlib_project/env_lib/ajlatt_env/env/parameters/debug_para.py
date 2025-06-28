#!opt/anaconda3/envs/lie_optimal/bin python
# _*_ coding: utf-8 _*_
'''
@File  : debug_para.py
@Author: Dongming Wang
@Email : dongming.wang@email.ucr.edu
@Date  : 10/12/2024
@Time  : 12:57:37
@Info  : Parameters used in debug, ssh debug or not
'''

import argparse

def get_debug_config():
    """
    Function: get_debug_para
    Description: TBC
    """

    parser = argparse.ArgumentParser(description="debug mode parameters")
    parser.add_argument('--ssh_debug', type=bool, default=False)
    # parser.add_argument('--ssh_debug', type=bool, default=True)
    return parser


