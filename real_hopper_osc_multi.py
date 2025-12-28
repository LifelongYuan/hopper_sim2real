import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import time
import numpy as np
from real_hopper_torch_utils import *
from hopper_config import HopperFlatCfg
import torch
from hopper_config import NUM_LEGS
PI=torch.pi

class RefMotionGenerator:
    def __init__(self,
                freegait_config:HopperFlatCfg.FreeGaitConfig,
                command_config:HopperFlatCfg.commands,
                sim_timestep,
                num_envs,
                device) -> None:

        self._sim_time_step=sim_timestep
        self.cfg = freegait_config
        self.command_cfg = command_config
        self._num_envs=num_envs
        self.device = device

        self.ref_quat = to_torch([0., 0., 0.,1], device=self.device).repeat((self._num_envs, 1))
        self.ref_yaw = torch.zeros([self._num_envs,1],device=self.device)
        self.z_vec = to_torch([0., 0., 1.], device=self.device).repeat((self._num_envs, 1))
        self.zero_vec = torch.zeros([1],device=self.device)


        self.ref_angle_vel = torch.zeros([self._num_envs,3],device=self.device)
        self.theta_change_now = torch.zeros([self._num_envs,NUM_LEGS],device=self.device)
        self.xy=torch.zeros([self._num_envs,NUM_LEGS,2],device=self.device)
        self.phase_header = torch.zeros([self._num_envs,1],device=self.device)
        # commmand buffer initialize
        # x_vel, y_vel, yaw_vel, phase_diff, heading 
        self.commands = torch.zeros(self._num_envs, self.command_cfg.num_commands, dtype=torch.float, device=self.device)
        self._precomputed_period = torch.zeros(self._num_envs, 1, dtype=torch.float, device=self.device)
        self.ref_roll = torch.zeros([self._num_envs,1],device=self.device)
        self.ref_pitch = torch.zeros([self._num_envs,1],device=self.device)

    def resample_initial_phase(self,env_ids):
        self.phase_header[env_ids,:] =  self.theta_change_now[env_ids,0].unsqueeze(dim=1)

    def update_commands(self,env_ids,command:torch.Tensor,duty=0.5):
        """
        reload free_gait config from commands.
        Return: init_phase after reset.
        """
        self.commands = command
        self.ref_angle_vel[env_ids,2]  = command[env_ids,2]       # dyaw
        if self.cfg.initialize_phase_schedule == "random":
            self.resample_initial_phase(env_ids)
            # single leg do not have gaits!
            # self._load_theta_from_diff(env_ids,self.phase_header,self.commands[:,3:5])
        elif self.cfg.initialize_phase_schedule == "zero":
            self.theta_change_now[env_ids] = 0
            self.phase_header[env_ids] = 0
        else:
            print("invalid initialize_phase_schedule")
            assert 0 # TODO bad error cast

        # normalize to [-pi,pi]    
        self.xy[:] = self._theta2xy(self.theta_change_now)
        self.theta_change_now = torch.atan2(self.xy[:,:,1],self.xy[:,:,0])
        self._precomputed_period[env_ids,:] = self.commands[env_ids,2].unsqueeze_(dim=1)
        # self.ref_roll[env_ids,0] = command[env_ids,8]       # roll_des
        # self.ref_pitch[env_ids,0] = command[env_ids,9]       # pitch_des
        return self._get_init_phase_started(env_ids)

    def _load_theta_from_diff(self,env_ids,head_phase,phase_diff):
        self.theta_change_now[env_ids,:] = head_phase[env_ids].repeat((1,NUM_LEGS))
        for index in range(NUM_LEGS-1):
            self.theta_change_now[env_ids,index+1] += phase_diff[env_ids,index+1] - phase_diff[env_ids,0]

    # update ref motion forward for self._sim_time_step.
    def update_osc(self):
        self._step_osc_iterations_fixed()
        self.ref_yaw[:,0]+= self.ref_angle_vel[:,2] * self._sim_time_step
        self.ref_yaw = normalize_angle(self.ref_yaw)
        self.ref_quat = quat_from_euler_xyz(self.ref_roll,self.ref_pitch,self.ref_yaw).squeeze(dim=1)


    def _step_osc_iterations_fixed(self):
        delta_theta = self._sim_time_step/self._precomputed_period * 2*torch.pi
        self.theta_change_now+=delta_theta
        self.xy[:] = self._theta2xy(self.theta_change_now)
        self.theta_change_now = torch.atan2(self.xy[:,:,1],self.xy[:,:,0])   # num_env*4

    def _theta2xy(self,theta_list:torch.Tensor):
        cos_theta = torch.cos(theta_list)  # num_env * 2
        sin_theta = torch.sin(theta_list)  # num_env *2
        xy = torch.dstack((cos_theta,sin_theta)) # num_env * 2 * 2
        # print("theta",xy.shape)
        return xy
    
    def reset(self,env_ids,init_yaw):
        self.ref_angle_vel[env_ids] = 0
        self.ref_yaw[env_ids,0] = init_yaw
        self.ref_quat[env_ids] = quat_from_euler_xyz(self.zero_vec,self.zero_vec,self.ref_yaw[env_ids]).squeeze(dim=1)

    def get_current_theta_change_now(self):
        return self.theta_change_now

    def get_current_leg_phase_xy(self):
        return self.xy      # num_env * NUM_LEGS * 2

    def get_current_ref_quat(self):
        return self.ref_quat
    
    def get_current_ref_yaw(self):
        return self.ref_yaw
    # The followings are private functions----------------------------------------------------------

    def _get_sim_step(self):
        return self._sim_time_step

    # can only be called after reset commands. This value is fixed with time. Only change when commands update
    def _get_init_phase_started(self,env_ids):
        return self.theta_change_now[env_ids,0]
