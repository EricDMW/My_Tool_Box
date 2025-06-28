from parameters import get_default_params

import matplotlib

from matplotlib import animation
from matplotlib import patches
from matplotlib import pyplot as plt
from gym import Wrapper
import numpy as np
from numpy import linalg as LA
from numpy import sqrt,trace
import os
from mpldatacursor import datacursor
import warnings

# Get default parameters
PARAMS = get_default_params()

class Display2D(Wrapper):
    def __init__(self, env, figID=0, skip=1, confidence=0.99, local_view=0):
        super(Display2D, self).__init__(env)
        
                
        # set the debug mode 
        if env.parser.ssh_debug:
            matplotlib.use('Agg')
        else:
            # Try to use TkAgg, but fall back to Agg if there's no display
            try:
                matplotlib.use('TkAgg')
            except:
                matplotlib.use('Agg')
        

        
        # figID = 0 : train, figID = 1 : test
        self.figID = figID 
         
        self.env_core = env.env
        self.im_size = self.env_core.parser.im_size
        self.bin = self.env_core.MAP.mapres
        self.mapmin = self.env_core.MAP.mapmin
        self.mapmax = self.env_core.MAP.mapmax
        self.mapres = self.env_core.MAP.mapres
        self.fig = plt.figure(self.figID)
        self.local_view = local_view
        if local_view:
            self.fig0 = plt.figure(self.figID+1)
            self.local_idx_map = [(1, 1), (1, 0), (1, 2), (0, 1), (2, 1)]
        self.n_frames = 0
        self.skip = skip
        self.c_cf = np.sqrt(-2*np.log(1-confidence))
        self.traj_num = 0

    def close(self):
        plt.close(self.fig)

    def step(self, action):
        # if type(self.env_core.targets) == list:
        #     target_true_pos = [self.env_core.targets.state[:2] for i in range(self.env_core.num_targets)]
        # else:
        #     target_true_pos = self.env_core.targets.state[:,:2]

        target_true_pos = self.env_core.target_true[0].state

        for i in range(self.env_core.num_agent):
            self.robot_traj[i][0].append(self.env_core.robot_true[i].state[0])
            self.robot_traj[i][1].append(self.env_core.robot_true[i].state[1])

        for i in range(self.env_core.num_target):
            self.target_traj[i][0].append(target_true_pos[0])
            self.target_traj[i][1].append(target_true_pos[1])
        
        for i in range(self.num_agent):
            self.robot_est_traj[i][0].append(self.robot_est[i].state[0])
            self.robot_est_traj[i][1].append(self.robot_est[i].state[1])

            # for j in range(self.num_target):
            self.target_est_traj[i][0].append(self.target_est[i][0].state[0])
            self.target_est_traj[i][1].append(self.target_est[i][0].state[1])

        return self.env.step(action)

    def render(self, record=False, batch_outputs=None,save_img=False):
        num_agent = self.env_core.num_agent
        
        # remark: record the true state
        robot_true = [self.env_core.robot_true[i].state
                      for i in range(num_agent)]
        robot_est = [self.env_core.robot_est[i].state
                     for i in range(num_agent)]
        robot_cov = [self.env_core.robot_est[i].cov
                     for i in range(num_agent)]

        num_targets = len(self.target_traj)

        target_true_pos = [ self.env_core.target_true[j].state for j in range(num_targets)]

        target_b_state = [
            [self.env_core.target_est[i][j].state for j in range(num_targets)] for i in range(num_agent)]
        target_cov = [[self.env_core.target_est[i][j].cov for j in range(num_targets)] for i in range(num_agent)]

        if self.n_frames % self.skip == 0:
            self.fig.clf()
            ax = self.fig.subplots()
            im = None
            if self.local_view:
                self.fig0.clf()
                if self.local_view == 1:
                    ax0 = self.fig0.subplots()
                elif self.local_view == 5:
                    ax0 = self.fig0.subplots(3, 3)
                    [[ax0[r][c].set_aspect('equal', 'box')
                      for r in range(3)] for c in range(3)]
                else:
                    raise ValueError('Invalid number of local_view.')

            robot_color = ['c','b','m','y','k','g']

            # MAP
            if self.env_core.MAP.visit_freq_map is not None:
                background_map = self.env_core.MAP.visit_freq_map.T
                if self.env_core.MAP.map is not None:
                    background_map += 2 * self.env_core.MAP.map
            else:
                if self.env_core.MAP.map is not None:
                    background_map = 2 * self.env_core.MAP.map
                else:
                    background_map = np.zeros(self.env_core.MAP.mapdim)

            im = ax.imshow(background_map, cmap='gray_r', origin='lower',
                        vmin=0, vmax=2, extent=[self.mapmin[0], self.mapmax[0],
                                                self.mapmin[1], self.mapmax[1]])


            # Plot robot true pose
            for i in range(num_agent):
                # ax.plot(robot_true[i][0], robot_true[i][1], marker=(3, 0, robot_true[i][2]/np.pi*180-90),
                #         markersize=10, linestyle='None', markerfacecolor='b',
                #         markeredgecolor='b')
                ax.plot(robot_true[i][0], robot_true[i][1], marker=(3, 0, robot_true[i][2]/np.pi*180-90),
                        markersize=10, linestyle='None', markerfacecolor=robot_color[i],
                        markeredgecolor=robot_color[i])

                # Plot robot probablilty Elliopse
                # Debug
                array_sum = np.sum(robot_cov[i])
                array_has_nan = np.isnan(array_sum)
                if array_has_nan:
                    chenck = 1
                    print('array_has_nan2')



                eig_val, eig_vec = LA.eig(robot_cov[i][:2, :2])
                belief_robot = patches.Ellipse(
                    (robot_est[i][0], robot_est[i][1]),
                    2*np.sqrt(eig_val[0])*self.c_cf,
                    2*np.sqrt(eig_val[1])*self.c_cf,
                    angle=180/np.pi*np.arctan2(np.real(eig_vec[0][1]),
                                               np.real(eig_vec[0][0])), fill=True, zorder=2,
                    facecolor=robot_color[i], alpha=0.5)
                ax.add_patch(belief_robot)

                # ax.plot(robot_true[0], robot_true[1], marker='>',
                #             markersize=10, linestyle='None', markerfacecolor='b',
                #             markeredgecolor='m')

                ax.plot(self.robot_traj[i][0],
                        self.robot_traj[i][1],  robot_color[i]+'.', markersize=2)

            for j in range(num_targets):
                # Plot target true trajectory
                ax.plot(self.target_traj[j][0],
                        self.target_traj[j][1], 'r.', markersize=2)
                if j == 0:
                    ax.plot(target_true_pos[j][0], target_true_pos[j][1], marker=(3, 0, target_true_pos[j][2]/np.pi*180-90),
                                markersize=10, linestyle='None', markerfacecolor='r',
                                markeredgecolor='r')
                else:
                    ax.plot(target_true_pos[j][0], target_true_pos[j][1], 'go', markersize=3)

                # Belief on target - Assuming that the first and the second dimension
                # of the target state vector correspond to xy-coordinate.
                for i in range(num_agent):
                    # for j in range(num_targets)
                        
                        if trace(target_cov[i][j])>=2*PARAMS['target_init_cov'][1]:
                            continue
                        eig_val, eig_vec = LA.eig(target_cov[i][j][:2, :2])
                        belief_target = patches.Ellipse(
                            (target_b_state[i][j][0], target_b_state[i][j][1]),
                            2*np.sqrt(eig_val[0])*self.c_cf,
                            2*np.sqrt(eig_val[1])*self.c_cf,
                            angle=180/np.pi*np.arctan2(np.real(eig_vec[0][1]),
                                                    np.real(eig_vec[0][0])), fill=True, zorder=2,
                            facecolor=robot_color[i], alpha=0.5)
                        ax.add_patch(belief_target)

                    

            # plot the arc for the sector
            for i in range(num_agent):
                sensor_arc = patches.Arc((robot_true[i][0], robot_true[i][1]), PARAMS['sensor_r_max']*2, PARAMS['sensor_r_max']*2,
                                         angle=robot_true[i][2]/np.pi*180, theta1=-PARAMS['fov']/2, theta2=PARAMS['fov']/2, facecolor='gray')
                ax.add_patch(sensor_arc)
                # plot the line for the sector
                ax.plot([robot_true[i][0], robot_true[i][0]+PARAMS['sensor_r_max']*np.cos(robot_true[i][2]+0.5*PARAMS['fov']/180.0*np.pi)],
                        [robot_true[i][1], robot_true[i][1]+PARAMS['sensor_r_max']*np.sin(robot_true[i][2]+0.5*PARAMS['fov']/180.0*np.pi)], 'k', linewidth=0.5)
                ax.plot([robot_true[i][0], robot_true[i][0]+PARAMS['sensor_r_max']*np.cos(robot_true[i][2]-0.5*PARAMS['fov']/180.0*np.pi)],
                        [robot_true[i][1], robot_true[i][1]+PARAMS['sensor_r_max']*np.sin(robot_true[i][2]-0.5*PARAMS['fov']/180.0*np.pi)], 'k', linewidth=0.5)



            # ax.text(self.mapmax[0]+1., self.mapmax[1]-5., 'v_target:%.2f'%np.sqrt(np.sum(self.env_core.targets[0].state[2:]**2)))
            # ax.text(self.mapmax[0]+1., self.mapmax[1]-10., 'v_agent:%.2f'%self.env_core.agent.vw[0])
            # ax.set_xlim((self.mapmin[0], self.mapmax[0]))
            # ax.set_ylim((self.mapmin[1], self.mapmax[1]))
            
            # plot communication
            for i in range(num_agent):
                for j in range(num_agent):
                    if i == j: 
                        continue
                    else:
                        if self.com_plot[i,j]:
                            x = np.array([robot_true[i][0],robot_true[j][0] ])
                            y = np.array([robot_true[i][1],robot_true[j][1] ])
                            ax.plot(x,y,color='b',linestyle='dashed')

            

            # ax.set_xlim((0, 120))
            # ax.set_ylim((0, 100))
            # ax.set_xlim((0, 40))
            # ax.set_ylim((0, 40))

            ax.set_xlim((self.mapmin[0], self.mapmax[0]))
            ax.set_ylim((self.mapmin[1], self.mapmax[1]))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

            if save_img:
                # matplotlib.pyplot.axis('off')
                # print()
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                ax.set_title('')
                plt.grid(True)
            else:
                ax.set_title("Trajectory %d" % self.traj_num)
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                ax.grid(True)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

            # ax.set_title("Trajectory %d" % self.traj_num)
            ax.set_aspect('equal', 'box')

            if self.local_view == 1:
                local_mapmin = np.array([-self.im_size/2*self.mapres[0], 0.0])
                ax0.imshow(
                    np.reshape(self.env_core.local_map[0], (self.im_size, self.im_size)),
                    cmap='gray_r', origin='lower', vmin=-1, vmax=1,
                    extent=[local_mapmin[0], -local_mapmin[0],
                            0.0, -local_mapmin[0]*2])
            elif self.local_view == 5:
                local_mapmin = np.array([-self.im_size/2*self.mapres[0], 0.0])
                [ax0[self.local_idx_map[j][0]][self.local_idx_map[j][1]].imshow(
                    np.reshape(self.env_core.local_map[j], (self.im_size, self.im_size)),
                    cmap='gray_r', origin='lower', vmin=-1, vmax=1,
                    extent=[local_mapmin[0], -local_mapmin[0],
                            0.0, -local_mapmin[0]*2]) for j in range(self.local_view)]
            if not record:
                plt.draw()
                plt.pause(0.0001)

        self.n_frames += 1

        
        # Now we can save it to a numpy array.
        # data = np.fromstring(self.fig.canvas.tostring_rgb(),
        #                      dtype=np.uint8, sep='')
        # data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        # Use the newer buffer_rgba() method instead of deprecated tostring_rgb()
        
        # Robust buffer capture with error handling
        data = self._capture_canvas_buffer()
        
        return data
    
    def _capture_canvas_buffer(self):
        """
        Robustly capture canvas buffer with multiple fallback strategies.
        Returns a properly shaped RGB array or a fallback image.
        """
        try:
            # Ensure canvas is ready
            self.fig.canvas.draw()
            
            # Capture buffer
            data = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            
            # Get canvas dimensions
            width, height = self.fig.canvas.get_width_height()
            
            # Strategy 1: Direct reshape if sizes match
            expected_size = width * height * 4
            if len(data) == expected_size:
                return self._reshape_to_rgb(data, height, width)
            
            # Strategy 2: Try to find valid dimensions
            return self._find_valid_dimensions(data, width, height)
            
        except Exception:
            # Strategy 3: Return fallback image
            return self._create_fallback_image()
    
    def _reshape_to_rgb(self, data, height, width):
        """Reshape RGBA data to RGB format."""
        try:
            data = data.reshape((height, width, 4))
            return data[:, :, :3]  # Drop alpha channel
        except ValueError:
            return self._create_fallback_image()
    
    def _find_valid_dimensions(self, data, original_width, original_height):
        """Find valid dimensions for the buffer data."""
        total_pixels = len(data) // 4
        
        if total_pixels * 4 != len(data):
            return self._create_fallback_image()
        
        # Try common aspect ratios
        aspect_ratio = original_width / original_height if original_height > 0 else 1.0
        
        # Strategy 2a: Try to maintain aspect ratio
        for scale_factor in [0.5, 1.0, 1.5, 2.0]:
            test_width = int(original_width * scale_factor)
            test_height = total_pixels // test_width
            if test_width * test_height == total_pixels:
                try:
                    return self._reshape_to_rgb(data, test_height, test_width)
                except ValueError:
                    continue
        
        # Strategy 2b: Try square dimensions
        side_length = int(np.sqrt(total_pixels))
        if side_length * side_length == total_pixels:
            try:
                return self._reshape_to_rgb(data, side_length, side_length)
            except ValueError:
                pass
        
        # Strategy 2c: Try to pad/crop to original dimensions
        if len(data) >= original_width * original_height * 4:
            # Crop to fit
            target_size = original_width * original_height * 4
            data = data[:target_size]
            return self._reshape_to_rgb(data, original_height, original_width)
        else:
            # Pad with zeros
            target_size = original_width * original_height * 4
            padded_data = np.zeros(target_size, dtype=np.uint8)
            padded_data[:len(data)] = data
            return self._reshape_to_rgb(padded_data, original_height, original_width)
    
    def _create_fallback_image(self):
        """Create a fallback image when buffer capture fails."""
        try:
            width, height = self.fig.canvas.get_width_height()
        except:
            width, height = 800, 600  # Default fallback size
        
        # Create a simple gradient fallback image
        fallback = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a subtle gradient to make it look intentional
        for i in range(3):
            fallback[:, :, i] = np.linspace(50, 100, width, dtype=np.uint8)[None, :]
        
        return fallback

    def reset(self, **kwargs):
        episode_num = PARAMS['episode_num']

        self.traj_num += 1
        self.robot_traj = [[[], []] for i in range(self.env_core.num_agent)]
        self.target_traj = [[[], []] for i in range(self.env_core.num_target)]
        
        self.robot_est_traj = [ [[], []] for i in range(self.num_agent)]
        self.target_est_traj = [ [[], []] for i in range(self.num_agent)]
        
        # self.robot_rmse = [ [[] for i in range(self.env_core.num_agent)] for j in range(episode_num) ] 
        # self.target_rmse = [ [[] for i in range(self.env_core.num_agent)] for j in range(episode_num) ] 
        
        # self.robot_obs_num_list = [[] for i in range(self.num_agent)]
        # self.robot_cov_trace_list = [[] for i in range(self.num_agent)]
        # self.target_cov_trace_list = [
        #     [[] for i in range(self.num_target)] for i in range(self.num_agent)]



        return self.env.reset(**kwargs)

    def record_trace(self,i_episode,T_steps,sampling_period,end_time):
        # RMSE calculation
        t = np.arange(0.0, T_steps*sampling_period, sampling_period)

        for i in range(self.num_agent):
            #TODO: Check the record data here
            # err = np.zeros((2,len(t)))
            err = np.zeros((2,end_time))
            # x error
            err[0,:end_time] = np.asarray(self.robot_est_traj[i][0])\
                 - np.asarray(self.robot_traj[i][0])
            # y error
            err[1,:end_time] = np.asarray(self.robot_est_traj[i][1])\
                 - np.asarray(self.robot_traj[i][1])

            RMSE = err.T@err
            self.robot_rmse[i_episode][i] = sqrt(np.diag(RMSE))

            for j in range(self.env_core.num_target):
                #TODO: check the record data here
                # err = np.zeros((2,len(t)))
                err = np.zeros((2,end_time))
                # x error
                err[0,:end_time] = np.asarray(self.target_est_traj[i][0])\
                    - np.asarray(self.target_traj[0][0])
                # y error
                err[1,:end_time] = np.asarray(self.target_est_traj[i][1])\
                    - np.asarray(self.target_traj[0][1])

                RMSE = err.T@err
                self.target_rmse[i_episode][i] = sqrt(np.diag(RMSE))
                

    def plot_all(self):
        # Monte carlo data calculation
        self.robot_rmse_np = np.array(self.robot_rmse)
        self.target_rmse_np = np.array(self.target_rmse)

        self.robot_cov_trace_np = np.array(self.robot_cov_trace_list)
        self.target_cov_trace_np = np.array(self.target_cov_trace_list)

        # for i in range(self.num_agent):
        robot_rmse_ave = np.sum(self.robot_rmse_np,axis = 0)/self.episode_num
        target_rmse_ave = np.sum(self.target_rmse_np,axis = 0)/self.episode_num

        robot_cov_trace_ave = np.sum(self.robot_cov_trace_np,axis = 0)/self.episode_num
        target_cov_trace_ave = np.sum(self.target_cov_trace_np,axis = 0)/self.episode_num
        end_time = len(robot_cov_trace_ave[0])
        
        t = np.arange(0.0, self.T_steps*self.sampling_period, self.sampling_period)

        plt.figure(1)
        print('robot cov',end='=  ')
        ax1 = plt.subplot(411)
        for i in range(self.num_agent):
            lines = plt.plot(t[:end_time], robot_cov_trace_ave[i], label='Robot'+str(i))
            print(round(np.average(robot_cov_trace_ave[i]), 3),end= ' ')
            
        print()
        ax1.set_title('robot cov')
        ax1.legend()

        # share x only
        print('target cov',end='=  ')
        ax2 = plt.subplot(412)
        for i in range(self.num_agent):
            # for j in range(self.num_target):
            plt.plot(t[:end_time], target_cov_trace_ave[i], label='Robot_T'+str(i))
            print(round(np.average(target_cov_trace_ave[i]), 3),end= ' ')
        print()
        ax2.set_title('target cov')
        ax2.legend()
        

        # RMSE plot
        print('Robot RMSE',end='=  ')
        ax3 = plt.subplot(413)
        ax3.set_title('Robot RMSE')
        for i in range(self.num_agent):
            plt.plot(t, robot_rmse_ave[i], label='Robot'+str(i))
            print(round(np.average(robot_rmse_ave[i]), 3),end= ' ')

            # plt.setp(ax3.get_xticklabels())
        ax3.legend()
        print()

        print('Target RMSE',end='=  ')
        ax4 = plt.subplot(414)
        ax4.set_title('Target RMSE')
        for i in range(self.num_agent):
            # make these tick labels invisible
            # plt.setp(ax4.get_xticklabels())
            # for j in range(self.env_core.num_target):
            plt.plot(t, target_rmse_ave[i], label='Target from Robot '+str(i))
            print(round(np.average(target_rmse_ave[i]), 3),end= ' ')

        ax4.legend()
        print()
        # datacursor()
        # datacursor(display='multiple', draggable=True)

        plt.show()   
        check = 1
        
class Video2D(Wrapper):
    def __init__(self, env, dirname='', skip=1, dpi=80, local_view=0):
        super(Video2D, self).__init__(env)
        self.local_view = local_view
        self.skip = skip
        self.moviewriter = animation.FFMpegWriter()
        fnum = np.random.randint(0, 1000)
        fname = os.path.join(dirname, 'train_%d.mp4' % fnum)
        self.moviewriter.setup(fig=env.fig, outfile=fname, dpi=dpi)
        if self.local_view:
            self.moviewriter0 = animation.FFMpegWriter()
            self.moviewriter0.setup(fig=env.fig0,
                                    outfile=os.path.join(
                                        dirname, 'train_%d_local.mp4' % fnum),
                                    dpi=dpi)
        self.n_frames = 0

    def render(self, *args, **kwargs):
        if self.n_frames % self.skip == 0:
            # if traj_num % self.skip == 0:
            self.env.render(record=True, *args, **kwargs)
        self.moviewriter.grab_frame()
        if self.local_view:
            self.moviewriter0.grab_frame()
        self.n_frames += 1

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def finish(self):
        self.moviewriter.finish()
        if self.local_view:
            self.moviewriter0.finish()
