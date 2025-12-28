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

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
from real_hopper import *
# from real_mc_nav import *
from hopper_config import *

from real_hopper_torch_utils import *
from real_hopper_gym_utils import *
# import lcm
# from lcm_config.python.rl_recorder import *
from rsl_rl.modules import ActorCritic,ActorCriticVAE
class logger_config:
    EXPORT_POLICY=True
    RECORD_FRAMES=False
    robot_index=0
    joint_index=1

        
class Player():
    def __init__(self,args):
        if args.cfg_dir=="local":
            env_cfg = HopperFlatCfg()
            train_cfg=HopperFlatCfgPPO()
            env_cfg,train_cfg=read_custom_cfg(env_cfg,train_cfg,os.path.dirname(args.checkpoint))
        elif args.cfg_dir=="":
            # load default cfg
            env_cfg = HopperFlatCfg()
            train_cfg=HopperFlatCfgPPO()
        else:
            env_cfg = HopperFlatCfg()
            train_cfg=HopperFlatCfgPPO()
            env_cfg,train_cfg=read_custom_cfg(env_cfg,train_cfg,args.cfg_dir)
    # prepare environment
        env = make_env(class_obj=RealHopper, args=args, env_cfg=env_cfg)
        self.obs = env.get_observations()
    # load policy
        self.env = env
        policy_cfg = class_to_dict(train_cfg)["policy"]

        _,sp=self.wrap_obs_infer(self.obs)
        pri_sp = None
        num_obs = sum(sp)
        if pri_sp is not None:
            num_critic_obs = sum(pri_sp)
        else:
            num_critic_obs = sum(sp)
            pri_sp=sp
        num_critic_obs = 390 # TODO
        self.actor_critic: ActorCriticVAE = ActorCriticVAE( 
                                                        num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        separate=sp,
                                                        critic_separate=pri_sp,
                                                        **policy_cfg).to(env.device)
        
        # self.load_actor_only(path=args.checkpoint)
        # self.load_jit(path=args.checkpoint)
        self.load(path=args.checkpoint)
        self.policy = self.actor_critic.act_inference
        self.critic = self.actor_critic.evaluate
        self.current_step_count = 0
        self.state_flag = 0
        self.actions = self.inference_normal(self.obs)

    # start = time.time()
    def load_jit(self,path):
        actor_path = path+"/policy_1_actor.pt"
        vae_path = path+"/policy_1_vae.pt"
        self.actor = torch.jit.load(actor_path)
        self.vae = torch.jit.load(vae_path)

    def load(self,path):
        loaded_dict = torch.load(path+"/policy_infer.pt")
        self.actor_critic.actor.load_state_dict(loaded_dict['actor_state_dict'])
        self.actor_critic.vae_est.load_state_dict(loaded_dict['vae_est_state_dict'])

    def wrap_obs_infer(self,obs):
        obs_hist=obs["obs_hist"]
        obs_current=obs["obs_current"]
        total = torch.cat((obs_hist,obs_current),dim=1)
        separate_point=[obs_hist.shape[1],obs_current.shape[1],3]
        return total,separate_point
    
    def inference(self,obs_dict):
        obs_his=obs_dict["obs_hist"]
        obs_current=obs_dict["obs_current"]
        z=self.vae.encode(obs_his)
        out=torch.concat([z[0],z[2],obs_current],dim=1)
        mean = self.actor(out)
        # print("vel",z[2])
        return mean
    
    def inference_normal(self,obs_dict):
        obs_his=obs_dict["obs_hist"]
        obs_current=obs_dict["obs_current"]
        # print("obs_hist",obs_his)
        z=self.actor_critic.vae_est.encode(obs_his)
        # print("z[0]",z[0])
        # z[2][:,-1] =0
        out=torch.concat([z[0],z[2],obs_current],dim=1)
        mean = self.actor_critic.actor(out)
        # print("vel",z[2])
        return mean
    
    def run_step(self,is_walking):
        with torch.inference_mode():
            now = time.time()
            self.actions = self.inference_normal(self.obs)
            # print('inference time',now-time.time())
            self.obs = self.env.step(self.actions)
            # print(1/(time.time()-now))
                # print(value)

        # print(obs[0,0:58])
if __name__ == '__main__':
    args = get_args()
    player=Player(args)
    while 1:
        player.run_step(True)





