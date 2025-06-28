#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : test.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: RL_LLM
@Date   : 10/05/2024
@Time   : 18:43:19
@Infor  : Description of the script
"""

from env import make
from parameters import all_parsers
import torch
def main():
   #BUILD THE MAIN FUNCTON
   config = all_parsers.parse_args()
   config.map_name = 'obstacles05'
   env = make(config)
   env.reset()
   for _ in range(1000):
      action = torch.rand(4,2)
      env.step(action)
      env.render()

if __name__ == '__main__':
    main()