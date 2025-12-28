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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
import torch.nn.functional as F
from .vae import VanillaVAE
from .rnd import RNDNetwork
class ActorCriticVAE(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        vae_latent_dim=16,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticVAE, self).__init__()

        activation = get_activation(activation)
        sp=kwargs["separate"]
        # critic_sp = kwargs["critic_separate"]
        mlp_input_dim_a = vae_latent_dim + sp[1] + sp[2]

        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)


        self.vae_est = VanillaVAE(in_channels=sp[0],
                                out_channels=sp[1],
                                latent_dim=vae_latent_dim,
                                estimate_dim=sp[2])
        self.rnd = RNDNetwork(input_dim=mlp_input_dim_a, hidden_dim=256, output_dim=128)
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"VAE: {self.vae_est}")
        print(f"RND: {self.rnd}")
        self.cp=[0]
        for step in range(len(sp)):
            sum_p=0
            for idx in range(step+1):
                sum_p+=sp[idx]
            self.cp.append(sum_p)
        # Action noise: will act like a model parameter(weights), and will be updated from loss degrade.
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.xavier_normal(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        """
        Only used for recurrent network.
        """
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        """
        Return entropy: entropy_p(x) = -E(log p(x)), where x ~p(x)
        """
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):

        obs_his = observations[:,self.cp[0]:self.cp[1]]
        obs_current = observations[:,self.cp[1]:self.cp[2]]
        z=self.vae_est.encode(obs_his)
        # check if there is nan:
        if torch.isnan(z[0]).any():
            print("nan in z[0]")
            input("nan detected!")
            # make nan into 0
            z[0] = torch.where(torch.isnan(z[0]), torch.zeros_like(z[0]), z[0])
        if torch.isnan(z[1]).any():
            print("nan in z[1]")
            input("nan detected!")
            # make nan into 0
            z[1] = torch.where(torch.isnan(z[1]), torch.zeros_like(z[1]), z[1])
        if torch.isnan(z[2]).any():
            print("nan in z[2]")
            input("nan detected!")
            # make nan into 0
            z[2] = torch.where(torch.isnan(z[2]), torch.zeros_like(z[2]), z[2]) 
            
        out = torch.concat([z[0].detach(),z[2].detach(),obs_current],dim=1)
        # check if there is nan:
        if torch.isnan(out).any():
            print("nan in out")
            input("nan detected!")
            # make nan into 0
            out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
        mean = self.actor(out)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        """
        sample action and update distribution.
        """
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        """
        log probability of value under such distribution
        """
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        """
        directly get action from network(no sampling)
        """
        obs_his = observations[:,self.cp[0]:self.cp[1]]
        obs_current = observations[:,self.cp[1]:self.cp[2]]
        z=self.vae_est.encode(obs_his)
        # only use mean
        out = torch.concat([z[0].detach(),z[2].detach(),obs_current],dim=1)
        mean = self.actor(out)
        return mean

    def evaluate(self, critic_observations,**kwargs):
        """
        Return the critic value from critic network.
        """
        value = self.critic(critic_observations)
        return value
        
    def rnd_inference(self,observations):
        obs_his = observations[:,self.cp[0]:self.cp[1]]
        obs_current = observations[:,self.cp[1]:self.cp[2]]
        z=self.vae_est.encode(obs_his)
        # only use mean
        out = torch.concat([z[0].detach(),z[2].detach(),obs_current],dim=1)
        prediction_error,_ = self.rnd(out)
        return prediction_error
    
    def vae_latent(self,observations):
        obs_his = observations[:,self.cp[0]:self.cp[1]]
        z=self.vae_est.encode(obs_his)
        return z
    
    def vae_generate(self,observations):
        obs_his = observations[:,self.cp[0]:self.cp[1]]
        recons=self.vae_est.generate(obs_his)
        return recons
    
    def vae_recon_deterministic(self,observations):
        obs_his = observations[:,self.cp[0]:self.cp[1]]
        recons=self.vae_est.generate_deterministic(obs_his)
        return recons
    
    def vae_loss(self,observations,next_obs,**kwargs):
        obs_his = observations[:,self.cp[0]:self.cp[1]]
        # obs_current = observations[:,self.cp[1]:self.cp[2]]
        # # mask obs_current: action:
        # obs_current[:,-12:] = 0
        obs_next = next_obs[:,self.cp[1]:self.cp[2]]
        obs_next[:,6:9] = 0   # mask actions
        real_vel = observations[:,self.cp[2]:self.cp[3]]
        recons = self.vae_est.generate(obs_his)
        z=self.vae_est.encode(obs_his)
        loss = self.vae_est.loss_function(recons,
                                        obs_next,
                                            z[0],
                                            z[1],
                                            real_vel,
                                            z[2])
        return loss

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
