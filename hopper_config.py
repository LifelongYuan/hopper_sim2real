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

import torch
NUM_JOINT = 3
NUM_LEGS = 1
CONST_OFFSET = 0.52
class HopperFlatCfg() :
    class env():
        num_envs = 4096
        # history_length=1 means only using current observation
        history_length = 12
        num_observations = (3+3+3+2+4+3)*history_length
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = NUM_JOINT
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        motor_max_change_per_step=0.4
        enable_filter=False

    class ref_motion():
        enable_loading = False
        file_path ="ref_motion_walk.pkl"
        frame_dt = 1/1000
        clear_init_pos = True
        clear_init_ori = True

    class init_state():
        pos = [0.0, 0.0, 0.45] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        motor_directions=[1]*NUM_JOINT
        # dof order:
    # ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
    # 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 
    # 'waist_pitch_joint', 'waist_roll_joint', 'waist_yaw_joint', 
    # 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 
    # 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']

        default_joint_angles = { # totally 23 dof_pos
            "Leg_Joint_Roll":0,
            "Leg_Joint_Pitch":0,
            "Leg_Joint_Shift":0.15
        }

    class termination():
        enable_early_termination=False
        enable_roll_pitch_termination=False
        distance_bound =2.5
        yaw_bound =torch.pi/4

    class video_logger():
        sampled_env_id = 0
        enable_video_logger = True
        video_log_interval = 50
        video_length_in_sec = 3
        env_dt = 0.02
        canvas_size = [360,240]

    class commands():             
        curriculum = False
        max_curriculum = 1.
        num_commands = 3 
        # lin_vel_x, lin_vel_y, ang_vel_yaw, [phase*2],period,height,walking_height,roll_des,pitch_des (8+2=10)
        resampling_time = 5. # time before command are changed[s]
        # "random"  "all" "gamepad" "route"
        resampling_schedule="random"
        vel_update_persent=0.8
        resample_by_rewards=False
        max_gait_sample_counts=500
        random_init_yaw=True
        disable_low_speed_gait=False

        route_dim = 3 # [delta_x,delta_y,delta_yaw]
        route_max_value = [10,10,torch.pi]  # max destination delta value
        route_max_vel = [1.0,0.5,0.7]
        tracking_kp=[0.15,0.15,1]

        class ranges:
            lin_vel_x = [-0.2,0.2 ] # min max [m/s]
            step_delta_lin_vel_x = 0.1
            max_lin_vel_x = 1.5

            lin_vel_y = [-0.2, 0.2]   # min max [m/s]
            step_delta_lin_vel_y = 0.1
            max_lin_vel_y = 1.0

            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
            step_delta_ang_vel=0.1
            max_ang_vel_yaw=1.5

            period = [0.3,0.4] # min max [s]
            swing_height = [0.08,0.15]
            walking_height = [0.9,1.0]
            roll = [-0.0,0.0]
            pitch=[-0.2,0.2]
            duty_ratio = [0.5,0.6]
            # phase_diff list commands can be found in free_gait config

    class record_testing():
        # min,max,resolution
        record_names=["lin_vel_x","period","swing_height","gait_num"]
        record_path_root="test_logs"
        verbose=True
        save_interval=1000  # TODO: remain unused
        test_max_episode_length_s=5
        record_ids=[0,7,8]
        lin_vel_x=[-1.0, 1.0,0.2]
        lin_vel_y=[0, 0,0] 
        ang_vel_yaw=[0, 0,0]
        walking_height = [0.25,0.35,0.02]
        period=[0.4,0.8,0.1]
        swing_height= [0.07,0.15,0.01]
        gait_num=[0,2,1]

    class viewer():
        ref_env = 0
        pos = [4, 2, 2]  # [m]
        lookat = [3., 2, 1.]  # [m]
    class control():
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {"Roll":20,"Pitch":20,"Shift":200}
        damping = {"Roll":0.5,"Pitch":0.5,"Shift":3}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        enable_interpolation = False
        add_delay = True
        delay_prob = 0.1
        delay_bound = [0.0, 1.0/50.0]

    class terrain():
        mesh_type = 'plane'
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 20 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 2 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols =  5# number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.3, 0.2, 0.2, 0.2]
        # terrain_proportions = [0.0, 1,0, 0.0, 0.0, 0.0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class asset():
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/hopper/urdf/hopper.urdf'
        name = "super"
        foot_name = "Foot_Link"
        penalize_contacts_on = []
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.005
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        randomize_base_com = True
        com_displacement_range=[-0.03, 0.03]
        added_mass_range = [-0.5, 0.5]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.3

        randomize_base_inertia=False
        base_inertia_ratio_range = [0.8,1.2]
        randomize_all_link_mass=True
        link_mass_ratio=[0.8,1.2]
        randomize_all_link_interia=True
        link_inertia_ratio_range = [0.8,1.2]

        randomize_pdgains=True
        pd_ratio=[0.9,1.1]
        add_motor_offset = True
        motor_offset_range = [-0.05, 0.05]
        randomize_motor_friction=False
        motor_friction_range = [0.0,0.1]

        randomize_motor_damping=False
        motor_damping_range = [0.0,0.07]

        randomize_rp_offset=True
        rp_offset_range = [-0.05,0.05]

        randomize_motor_strength = True
        motor_strength_ratio_range = [0.8,1.2]
        
    class rewards:
        class scales:
            termination = -0.0
            # tasks
            tracking_lin_vel = 3.0   # 40.0
            # tracking_ang_vel = 0.5
            dof_pos_limits = -10
            tracking_contacts_force= -0.1 #-0.01
            penalize_nonflat_foot_ori = -0.0 
            tracking_pos = 0.0
            tracking_yaw = 0.0
            penalize_ang_vel_z = 0.0
            foot_z = 1.0
            body_z = 0.0
            foot_xy = 0.0
            orientation = 0.0
            dof_vel = -0.008 #-0.1 # velocity width limitation
            dof_acc = -2.5e-7
            action_rate = -0.03

            regulate_joint_pos = 0.0
            feet_distance = 0.0
            tracking_base_roll = 0.5
            tracking_base_pitch = 0.5

            tracking_contacts_vel = 0 #10
            tracking_swing_height = 0.0   # -0.2
            tracking_swing_xy =0.0     # 5
            penalty_ang_roll_pitch_vel=-0.0
            regulate_abad = -0.00

            # reference motion
            ref_dof_pos = 0.0  #5
            power_min = 0.0
            base_z = -0.000
            lin_vel_z = -0.0  # 3.0

            ang_vel_xy = -0.03
            ang_vel_z = -0.05

            dof_vel = -0.008

            torques = -0.00004
            torque_limits=0.
            torque_distribution = 0.0

            base_height = -0. 
            feet_air_time = 0.
            
            feet_contact_forces=-0.00  # -0.015
            collision = -0.0  # -5.0
            stumble  = 0.0
            stumble_reflection = -0.0

            action_vel_rate= 0.0
            stand_still = -0.
            feet_impact_vel= -0.0 # 0.0


        feet_min_dist = 0.2
        feet_max_dist = 0.5
        power_window_width = 30
        regulate_sigma = 1
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma_lin_vel = 0.1**2  # 0.1**2  # tracking reward = exp(-error^2/sigma)
        regulate_abad_sigma = 0.08**2
        tracking_sigma_ang_vel = 0.25
        tracking_sigma_swing_height = 0.05**2
        tracking_sigma_swing_height_xy = 0.05**2
        tracking_sigma_dof_pos = 0.2**2
        tracking_sigma_dof_vel = 50.
        tracking_sigma_base_pos = 2.
        tracking_sigma_base_yaw=0.3
        tracking_sigma_base_roll=0.03**2
        tracking_sigma_base_pitch=0.03**2
        tracking_sigma_contacts_force = 20**2
        tracking_sigma_contacts_vel = 0.5
        tracking_sigma_penalty_ang_roll_pitch_vel=0.05
        base_z_sigma=0.01**2
        lin_vel_z_sigma=0.05
        power_sigma = 1000.
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.7
        base_height_target = 0.45        # print("yaw_error",normalize_angle(yaw))
        max_contact_force = 60. # forces above this value are penalized
        feet_impact_vel_sigma = 100.
        action_rate_weights=[1]*NUM_JOINT
        action_rate_sigma = 200
        action_vel_rate_sigma = 700

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            quat = 1.0 # important information
            cmd_lin_vel = 2.0
            cmd_ang_vel = 0.25
            cmd_phase = 0.5
            cmd_head = 0.0
            cmd_period = 2.0
            cmd_swing_height=2.0
            cmd_walking_height=0.4
            cmd_roll = 1.0
            cmd_pitch = 1.0
            dof_pos = 1.0
            dof_vel = 0.00
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.015
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.5
            gravity = 0.05
            rpy=0.01
            height_measurements = 0.1

    class FreeGaitConfig:
        # osc config
        alpha=50
        mu=1
        beta=0.5
        b=50
        dt=0.001
        # trajectory config
        k=1
        omega=0.0         # angular velocity     
        base_length=0.37 # TODO

        # phase_diff=np.array([0.0,torch.pi,torch.pi,0.0])  # define the phase diff between legs
        walking_height=0.3  # TODO
        repeat_times=1
        # random zero
        initialize_phase_schedule="random"
        upper_link_length=0.211  # TODO
        lower_link_length=0.20
        pace_offset=0.00

        phase_selections=[
                                            [0,torch.pi],                                  # trot 
                                            # [0,0,torch.pi,torch.pi],                                    # bounce
                                            # # [0,3*torch.pi/4,3*torch.pi/4,0], # half half trot
                                            # # [0,0,3*torch.pi/4,3*torch.pi/4],   # half bounce
                                            # [0,torch.pi/2,torch.pi,1.5*torch.pi],    # 3-walk
                                            # [0,2*torch.pi/3,4*torch.pi/3,0],            # 4-walk
                                            # [0,torch.pi,0,torch.pi]   # pace
                                            ]                           
        #                                   # trot 
                  
        #                                     [0,torch.pi,0,torch.pi],        
        # phase_selections=[[0,torch.pi,torch.pi,0],                                  # trot 
        #                                     [0,torch.pi,0,torch.pi],                                    
        #                                     [0,0,torch.pi,torch.pi]]
        # phase_selections=[[0,torch.pi,torch.pi,0]]
        # phase_selections=[[torch.pi/2,torch.pi,torch.pi/2,torch.pi],[torch.pi/2,torch.pi,torch.pi/2,torch.pi],[torch.pi/2,torch.pi,torch.pi/2,torch.pi]]
        # phase_selections=[[0,torch.pi,0,torch.pi],[0,torch.pi,0,torch.pi],[0,torch.pi,0,torch.pi]]
    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.03  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class HopperFlatCfgPPO( ):
    seed = 1
    runner_class_name = 'OnPolicyRunnerVAE'
    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
    class algorithm():
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.e-5 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner():
        policy_class_name = 'ActorCriticVAE'
        algorithm_class_name = 'PPOVAE'
        num_steps_per_env = 24 # per iteration
        max_iterations = 300000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'hopper'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

  
