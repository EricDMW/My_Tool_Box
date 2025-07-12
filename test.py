# import envlib_project.env_lib.linemsg_env as linemsg
# import envlib_project.env_lib.pistonball_env as pistonball
# import envlib_project.env_lib.wireless_comm_env as wireless_comm
# import envlib_project.env_lib.kos_env as kos

import env_lib
import toolkit
import numpy as np
import matplotlib.pyplot as plt 
import torch
import gymnasium as gym
import warnings
import time
# warnings.filterwarnings('error')



# require above import as
env = gym.make('LineMsg-v0')
env = gym.make('Pistonball-v0')
env = gym.make('WirelessComm-v1')
env = gym.make('KuramotoOscillator-v1') 

# # directly use the class
# env = linemsg.linemsg_env.LineMsgEnv(num_agents=5)
# env = pistonball.pistonball_env.PistonballEnv(n_pistons=5)

# require to pip install -e . under env_lib folder
env = env_lib.LineMsgEnv(num_agents=5)
env = env_lib.PistonballEnv(n_pistons=5)
env = env_lib.KuramotoOscillatorEnv(n_oscillators=5)
env = env_lib.KuramotoOscillatorEnvTorch(n_oscillators=5)
env = env_lib.WirelessCommEnv(grid_x=5, grid_y=5)
 
env = env_lib.ajlatt_env(map_name='obstacles04',num_Robot=4)
env.reset()
start_time = time.time()
for _ in range(100):
    action = 0.1 * torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    k = env.step(action)
    print(k[1])
    env.render()
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

# x = np.arange(100)
# y = np.sin(x / 10)
# y_std = 0.1 + 0.1 * np.abs(np.cos(x / 10))
# plotkit.plot_shadow_curve(x, y, y_std, label='sin(x)', color='blue')
# plt.legend()
# plt.show()
# plt.savefig('sin.png')
# plt.close()