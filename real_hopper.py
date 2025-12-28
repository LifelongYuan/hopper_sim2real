# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import numpy as np
import math
import threading
import torch
from real_hopper_torch_utils import *
from real_hopper_osc_multi import *
from real_hopper_torch_utils import *
from real_hopper_gym_utils import *
from hopper_config import *
import lcm_interface
from scipy.spatial.transform import Rotation as R
from util.utilities import *
HACK_ACTION_SCALE = False
class RealHopper():
    cfg : HopperFlatCfg
    def __init__(self, cfg):
        self.cfg = cfg
        
        # force set to 1.
        self.cfg.env.num_envs=1
        self.cfg.domain_rand.randomize_pdgains=False
        self.cfg.commands.resampling_schedule ="random"
        self.cfg.terrain.mesh_type = "plane"
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.last_time = time.time()

        # cpu only for real_mc_standalone
        self.device = 'cpu'

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        self.num_dofs = NUM_JOINT
        self.dof_names = ['Leg_Joint_Roll',
                              'Leg_Joint_Pitch',
                              'Leg_Joint_Shift'
                            ]

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            # self.num_privileged_obs = self.num_obs

        self.extras = {}

        # initialize vectors
        self._recv_motor_angles=np.zeros(NUM_JOINT)
        self._recv_motor_vels=np.zeros(NUM_JOINT)
        self.package_counting = 0.0
        self.last_package_counting = 0.0
        self._tweaked_actions=np.zeros(NUM_JOINT)
        # self.motor_direction_correction=np.ones(12)
        self.motor_direction_correction=np.array(self.cfg.init_state.motor_directions)
        self.motor_directions_torch = to_torch(self.cfg.init_state.motor_directions,device=self.device)
        self.gym_motor_directions_torch = to_torch([1,1,1],device=self.device)
        self.gym_motor_directions = np.array([1,1,1])
        self.final_actions = np.zeros(NUM_JOINT)
        self.intered_actions = np.zeros(NUM_JOINT)
        self._last_final_actions = np.zeros(NUM_JOINT)
        self.filtered_actions = None
        # self.cos_action = np.zeros(12)
        # self.step_num=0
        self._init_buffers()
        self.init_done = True
        self._init_ref_motion()
        self.time_per_render = self.cfg.sim.dt * self.cfg.control.decimation
        # self._ref_motion = refmotion()
        self._lcm_interface = lcm_interface.robot_lcm(
            publish_freq=50,
            ttl=1,
            pub_topic_name="hopper_cmd",
            sub_topic_name_dict={"data":"hopper_data",
                                "imu":"hopper_imu",
                                "gamepad":"gamepad"},
            joint_state_ros_pub=False)
        # self._gameped_interface = lcm_interface.gamepad_lcm()
        env_ids =torch.ones([self.num_envs]).bool().nonzero(as_tuple=False).flatten()
        # first receive observation before reset ref motion.
        self._state_flag = 0
        self.unsafe_flag = False
        self.decode_observation()

        self._reset_ref_motion(env_ids)
## init camera related settings:
        self.apply_nav = False
        self._init_thread()
        self.s=time.time()
        
    def get_observations(self):
        obs_dict = {"obs_hist":self.obs_buf,"obs_current":self.obs_current}
        return obs_dict

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.cfg.sim.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.commands.resampling_schedule == "all":
            self.record_testing_config_dict = class_to_dict(self.cfg.record_testing)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    # prevent calling reset_idx from parent class.
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
    def _init_thread(self):
        self._lcm_interface.start_sending()
        self._lcm_interface.start_recving()
        # self._gameped_interface.start_recving()
        self.main_thread_locker = threading.Lock()

    def step(self,actions):
        # self.main_thread_locker.acquire()
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions)
        # if int(self._state_flag) !=  1:
        #     self.actions*=0
    
        # prepare stage: refresh initial yaw
        if int(self._state_flag) == 0:
            self.init_base_yaw = normalize_angle(get_euler_xyz(self.base_quat)[2])
            env_ids =torch.ones([self.num_envs]).bool().nonzero(as_tuple=False).flatten()
            self._reset_ref_motion(env_ids)
            print("current base yaw when reseting",self.init_base_yaw)

        if self.action_filter is not None:
            self.filtered_actions = self.action_filter.filter(self.actions.clone())
        else:
            self.filtered_actions=self.actions

        self.main_thread_locker.acquire()
        # # print("final_actions",self.final_actions)
        # sim body-> real body-> foot
        if HACK_ACTION_SCALE:
            action_scale = to_torch([self.cfg.control.action_scale/3.6,
                                     self.cfg.control.action_scale/1.0,
                                     self.cfg.control.action_scale/3.6],device=self.device)
        else:
            action_scale = self.cfg.control.action_scale
        full_action = self.filtered_actions * self.cfg.control.action_scale + self.default_dof_pos
        print("full_action",full_action)
        foot_pos = fk_simplified(full_action[0,0],full_action[0,1],full_action[0,2] - CONST_OFFSET)
        # print("foot_pos",foot_pos)
        # rot = R.from_matrix(foot2body().T*realbody2simbody().T)
        # foot_pos = rot.apply(foot_pos)
        # print("foot_pos",foot_pos)
        # foot_pos/=2
        # TODO: double check the range!
        print(foot_pos[2])
        # foot_pos[2] = np.clip(foot_pos[2],0.25,0.55)
        kp = 10.0
        kd = 0.2
        print(foot_pos)
        
        # Set LCM command message
        # full_action is already in joint space, so use it directly for q_des
        if self.unsafe_flag==False:
            self._lcm_interface.c2r_cmd_msg.q_des = list(full_action[0].cpu().numpy())
            self._lcm_interface.c2r_cmd_msg.qd_des = [0.0, 0.0, 0.0]  # zero desired velocity
            self._lcm_interface.c2r_cmd_msg.kp_joint = [kp, kp, kp]
            self._lcm_interface.c2r_cmd_msg.kd_joint = [kd, kd, kd]
            self._lcm_interface.c2r_cmd_msg.tau_ff = [0.0, 0.0, 0.0]  # zero feedforward torque
        else:
            print("shut down!!")
            # Send zero gains for safety
            self._lcm_interface.c2r_cmd_msg.q_des = list(full_action[0].cpu().numpy())
            self._lcm_interface.c2r_cmd_msg.qd_des = [0.0, 0.0, 0.0]
            self._lcm_interface.c2r_cmd_msg.kp_joint = [0.0, 0.0, 0.0]
            self._lcm_interface.c2r_cmd_msg.kd_joint = [kd, kd, kd]
            self._lcm_interface.c2r_cmd_msg.tau_ff = [0.0, 0.0, 0.0]
        # print("spent:",time.time()-self.s)
        self.main_thread_locker.release()
        # print(self.final_actions[:])
        elapsed_time = time.time() - self.s
        time.sleep(max(0, 0.02 - elapsed_time))
        # time.sleep(0.02)
        if int(self._state_flag) ==  1:
            self._update_ref_motion()
        self._update_ref_motion_command(self._first_env)
        self.main_thread_locker.acquire()
        # s_cu = time.time()
        self.s = time.time()
        self.decode_observation()
        self.common_step_counter +=1
        self.main_thread_locker.release()
        # now = time.time()
        # print("decoding time2",time.time()-now)
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        obs_dict = {"obs_hist":self.obs_buf,"obs_current":self.obs_current}
        return obs_dict
    
    def roll_pitch_safety_check(self,r,p):
        if torch.abs(normalize_angle(r)) > 0.5 or torch.abs(normalize_angle(p)) > 0.5:
            self.unsafe_flag = True
            print("unsafe")

    def decode_observation(self):
        # rpy = np.array(self._lcm_interface.c2p_msg.rpy)
        self._state_flag = 1
        
        # Get joint positions and velocities directly from LCM hopper_data message
        joint_pos = np.array(list(self._lcm_interface.r2c_data_msg.q))
        joint_vel = np.array(list(self._lcm_interface.r2c_data_msg.qd))
        
        # Apply CONST_OFFSET to joint position (matching original code behavior)
        joint_pos[2] += CONST_OFFSET
        
        # Compute foot_pos from joint positions for debugging/compatibility
        joint_pos_for_fk = joint_pos.copy()
        joint_pos_for_fk[2] -= CONST_OFFSET  # Remove offset for FK computation
        foot_pos = fk_simplified(joint_pos_for_fk[0], joint_pos_for_fk[1], joint_pos_for_fk[2])
        
        # Apply coordinate transformation: foot -> real body -> sim body
        rot = R.from_matrix(realbody2simbody()*foot2body())
        foot_pos = rot.apply(foot_pos)
        
        # Verify joint_pos is valid (check for NaN)
        if np.sum(np.isnan(joint_pos)) != 0:
            print("NaN detected in joint_pos from LCM, resetting to zero")
            joint_pos = np.array([0.0, 0.0, 0.0])

        # Get IMU data from LCM hopper_imu message
        # LCM quat format: [w, x, y, z] or [x, y, z, w] - need to check
        # Based on the code, it seems the original format was wxyz, so assuming LCM is [w, x, y, z]
        ori_raw_imu = list(self._lcm_interface.r2c_imu_msg.quat)
        if len(ori_raw_imu) == 0 or all(x == 0 for x in ori_raw_imu):
            ori_raw_imu = [1, 0, 0, 0]
        
        # Convert from LCM quat format (assuming [w, x, y, z]) to scipy format [x, y, z, w]
        q_raw_imu = R.from_quat([ori_raw_imu[1], ori_raw_imu[2], ori_raw_imu[3], ori_raw_imu[0]])
        imu_raw_to_body = R.from_euler('z', 0) * R.from_euler('x', 1.57)
        rotated_quaternion = (imu_raw_to_body * q_raw_imu).as_quat()
        quat = [rotated_quaternion[0], rotated_quaternion[1], rotated_quaternion[2], rotated_quaternion[3]]
        q_imu_raw = to_torch(quat, device=self.device).unsqueeze(dim=0)
        r, p, y = get_euler_xyz(q_imu_raw)
        temp = r
        r = -p
        p = temp * 0

        # if self.common_step_counter > 50:
        #     self.roll_pitch_safety_check(r, p)
        print("roll_raw", normalize_angle(r))
        print("pitch_raw", normalize_angle(p))
        print("y_raw", normalize_angle(y))

        # Get angular velocity from LCM IMU message
        ang_vel = np.array(list(self._lcm_interface.r2c_imu_msg.gyro))
        rot = R.from_matrix(realbody2simbody())
        ang_vel = rot.apply(ang_vel)
        # print("ang_vel",ang_vel)
        # q = to_torch(quat,device=self.device).unsqueeze(dim=0)
        # r,p,y=get_euler_xyz(q)
        # r +=0.1
        # r*=0
        # p*=0
        # print("roll",normalize_angle(r))
        # print("pitch",normalize_angle(p))
        # print("y",normalize_angle(y))
        # print("ang_vel",ang_vel) 
        self._recv_orientation = quat_from_euler_xyz(normalize_angle(r),normalize_angle(p),normalize_angle(y))
        # self._tweak_motor_order_direction()
        # print("_recv_motor_angles", self._recv_motor_angles)
        # print("_recv_motor_vels", self._recv_motor_vels)
        # print(self._lcm_interface.c2p_msg.v_des)
        vx=0.0
        vy=0.0
        desired_period = 0.42
        # print(desired_height)
        self._recv_commands = [vx,vy]+ \
                              [desired_period]
        # print("_recv_commands", self._recv_commands)
        # print("actions",self.actions)
        # print("default_pos", self.default_dof_pos)
        # if (abs(self.package_counting - self.last_package_counting-0.001)>0.002):
        #     print("lost!")
        # self.last_package_counting = self.package_counting
        # turn numpy to tensors:
        self.base_quat = self._recv_orientation.view(1,-1)
        self.base_ang_vel = to_torch(ang_vel,device=self.device).view(1,-1)
        self.dof_pos = to_torch(joint_pos,device=self.device).view(1,-1)
        self.commands = to_torch(self._recv_commands,device=self.device).view(1,-1)
        self.obs_current = torch.cat((  
                                    self.base_ang_vel  * self.obs_scales.ang_vel,   # 3
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos*self.gym_motor_directions_torch,  # 12
                                    self.actions,   # 3reset_header
                                    self.current_ref_leg_phase_xy.flatten(start_dim=1),  # 2
                                    self.base_quat * self.obs_scales.quat,   # 4
                                    self.commands * self.commands_scale
                                    # self.place_holders          # 4
                                    ),dim=-1)
        
        # print("obs_current",self.obs_current)
        # print("current_ref_leg_phase_xy",self.current_ref_leg_phase_xy)
        if self._history_length > 1:
            self._wrap_observation_history()
            self.obs_buf = self.obs_buf_history.view(self.num_envs,-1)
        else:
            self.obs_buf = self.obs_current
        self.last_obs_current[:] = self.obs_current[:]
        # print(self.current_ref_leg_phase_xy.reshape(8))



    def _tweak_motor_order_direction(self):
        # isaac order:  LF LH RF RH     abduction +hip+ knee
        # cheetah software order: FR FL HR HL adduction + hip + knee
        # Get joint positions and velocities from LCM
        raw_angles = np.array(list(self._lcm_interface.r2c_data_msg.q)) * self.motor_direction_correction
        raw_vels = np.array(list(self._lcm_interface.r2c_data_msg.qd)) * self.motor_direction_correction
        # print(raw_angles)
        self._recv_motor_angles[0:3] = raw_angles[3:6]
        self._recv_motor_angles[3:6] = raw_angles[0:3]

        self._recv_motor_angles[6:9] = raw_angles[9:12]
        self._recv_motor_angles[9:12] = raw_angles[6:9]

        self._recv_motor_vels[0:3] = raw_vels[3:6]
        self._recv_motor_vels[3:6] = raw_vels[0:3]

        self._recv_motor_vels[6:9] = raw_vels[9:12]
        self._recv_motor_vels[9:12] = raw_vels[6:9]


        # change directions
        self._recv_motor_angles =  self._recv_motor_angles
        self._recv_motor_vels =  self._recv_motor_vels
    
    def _process_action_to_real(self,raw_actions):
        # raw_actions=self.actions.clone() *motor_direction
        # change residual to full

        full_action = raw_actions.numpy() * self.cfg.control.action_scale*self.gym_motor_directions + self.default_dof_pos.numpy()
        # tweak orders
        self._tweaked_actions = full_action
        # self._tweaked_actions[0:3] = full_action[0,3:6]
        # self._tweaked_actions[3:6] = full_action[0,0:3]

        # self._tweaked_actions[6:9] = full_action[0,9:12]
        # self._tweaked_actions[9:12] = full_action[0,6:9]

        # tweak directions
        self.final_actions = self._tweaked_actions * self.motor_direction_correction

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """

        self.init_base_yaw=torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        # history buffer added 
        self._history_length = self.cfg.env.history_length
        self.obs_current = torch.zeros(self.num_envs, self.num_obs//self._history_length, device=self.device, dtype=torch.float)
        self.last_obs_current = self.obs_current.clone()
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.filtered_actions =  torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_filtered_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.received_command = self.commands.clone()
        self.desired_route = torch.zeros(self.num_envs,self.cfg.commands.route_dim,dtype=torch.float,device=self.device,requires_grad=False)
        self.current_route = torch.zeros(self.num_envs,self.cfg.commands.route_dim,dtype=torch.float,device=self.device,requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.cmd_lin_vel, 
                                            self.obs_scales.cmd_lin_vel, 
                                            self.obs_scales.cmd_period
                                            ], device=self.device, requires_grad=False)
        self.ang_vel_error= torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        # hard code // to be improved.
        self.phase_selections = to_torch(self.cfg.FreeGaitConfig.phase_selections,device=self.device)
        self.place_holders = torch.zeros(self.num_envs,2, dtype=torch.float, device=self.device)
        self.duty_ratio = torch.zeros(self.num_envs,1, dtype=torch.float, device=self.device, requires_grad=False)

        self.current_ref_leg_phase_xy = torch.zeros(self.num_envs,1,2,dtype=torch.float, device=self.device)
        self.current_ref_yaw=torch.zeros(self.num_envs,1,dtype=torch.float,device=self.device)
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self._first_env=  torch.Tensor([1,0],device=self.device).bool().nonzero(as_tuple=False).flatten()

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    if self.cfg.domain_rand.randomize_pdgains:
                        self.p_gains[:,i] = torch_rand_float(lower=self.cfg.control.stiffness[dof_name]*self.cfg.domain_rand.pd_ratio[0],
                                                                                            upper=self.cfg.control.stiffness[dof_name]*self.cfg.domain_rand.pd_ratio[1],
                                                                                            shape=(self.num_envs,1),
                                                                                            device=self.device).squeeze(dim=1)
                        self.d_gains[:,i]=torch_rand_float(lower=self.cfg.control.damping[dof_name]*self.cfg.domain_rand.pd_ratio[0],
                                                                                            upper=self.cfg.control.damping[dof_name]*self.cfg.domain_rand.pd_ratio[1],
                                                                                            shape=(self.num_envs,1),
                                                                                            device=self.device).squeeze(dim=1)
                    else:
                        self.p_gains[:,i] = to_torch(self.cfg.control.stiffness[dof_name],device=self.device)
                        self.d_gains[:,i] = to_torch(self.cfg.control.damping[dof_name],device=self.device)
                found = True
            if not found:
                self.p_gains[:,i] = 0.
                self.d_gains[:,i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        if self._history_length > 1:
            self._init_observation_history_buffer()
            self.action_filter = None

    def _wrap_observation_history(self):
        self.obs_buf_history = self.obs_buf_history.roll(1,dims=1)
        self.obs_buf_history[:,0,:] = self.last_obs_current
        # print(self.obs_buf)

    def _init_observation_history_buffer(self):
        self.obs_buf_history = torch.zeros(self.num_envs,self._history_length,self.num_obs//self._history_length,device=self.device)

    def _update_ref_motion(self):
        self._ref_generator.update_osc()
        self.current_ref_leg_phase_xy = self._ref_generator.get_current_leg_phase_xy()
        self.current_ref_yaw = self._ref_generator.get_current_ref_yaw()


    def _update_ref_motion_command(self,env_ids):
        if self.cfg.commands.resampling_schedule =="gamepad":
            self._ref_generator.update_commands(env_ids,self.received_command,self.duty_ratio)

        else:
            self._ref_generator.update_commands(env_ids,self.commands,self.duty_ratio)

    def _init_ref_motion(self):
        self._ref_generator=RefMotionGenerator(self.cfg.FreeGaitConfig,
                                               self.cfg.commands,
                                               self.cfg.sim.dt*self.cfg.control.decimation,
                                               self.num_envs,
                                               self.device)

    def _interpolation_action(self,target,start,process):
        return start + (target-start) * process

    def _reset_ref_motion(self,env_ids):
        self._ref_generator.reset(env_ids,self.init_base_yaw[env_ids])
        if self.cfg.commands.resampling_schedule =="gamepad":
            self._ref_generator.update_commands(env_ids,self.received_command,self.duty_ratio).unsqueeze(dim=1)

        elif self.cfg.commands.resampling_schedule =="random" or self.cfg.commands.resampling_schedule =="route":
            self._ref_generator.update_commands(env_ids,self.commands,self.duty_ratio).unsqueeze(dim=1)
        else:
            print("invalid resampling_schedule")
            assert 0 # TODO bad error cast    # update commands within an episode, header will keep invariant.
        self.current_ref_leg_phase_xy[env_ids] = self._ref_generator.get_current_leg_phase_xy()[env_ids]
        self.current_ref_yaw[env_ids] = self._ref_generator.get_current_ref_yaw()[env_ids]

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs
