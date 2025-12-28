"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
import os
import torch
import numpy as np
import torch
import random
import copy
import argparse
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from real_hopper import *
def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        if args.loco_checkpoint is not None:
            env_cfg.loco_policy.path = args.loco_checkpoint
            
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def make_env(class_obj, args=None, env_cfg=None):
    # if no args passed get command line arguments
    if args is None:
        args = get_args()
    # override cfg from args (if specified)
    env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
    set_seed(args.seed)
    # parse sim params (convert to dict first)

    env = class_obj(cfg=env_cfg)
    return env

def make_alg_runner(env, name=None, args=None, train_cfg=None, log_root="default"):
    # if no args passed get command line arguments
    if args is None:
        args = get_args()
    # override cfg from args (if specified)
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)

    log_dir = None
    
    train_cfg_dict = class_to_dict(train_cfg)
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
    #save resume path before creating a new log_dir
    resume = train_cfg.runner.resume
    if resume:
        # load previously trained model
        resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
        runner.load(resume_path)
    return runner, train_cfg,log_dir
    
def get_load_path(root, load_run=-1, checkpoint=""):
    
    print("load_path",checkpoint)
    return checkpoint

def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
    arg_parser.add_argument("--checkpoint", dest="checkpoint", type=str, default="")
    arg_parser.add_argument("--resume", dest="resume", action="store_true", default=False)
    arg_parser.add_argument("--rl_device", dest="rl_device", type=str, default="cpu")
    arg_parser.add_argument("--enable_nav",dest="enable_nav",action="store_true",default=False)
    arg_parser.add_argument("--loco_checkpoint",dest="loco_checkpoint",type=str)
    arg_parser.add_argument("--cfg_dir",dest="cfg_dir",type=str,default="")

    args = arg_parser.parse_args()
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def config_roll_out2dict(class_name,dict_obj):
    """Roll out class to dict"""
    for attr in dir(class_name):
        if  not attr.startswith('__'):
            if isinstance(getattr(class_name, attr),type):
                class_dict={}
                dict_obj[str(attr)] = class_dict
                config_roll_out2dict(getattr(class_name, attr),class_dict)
            else:
                dict_obj[str(attr)] = getattr(class_name, attr)

def config_fill_in_from_dict(cfg,dict_obj):
    """Fill in class from dict"""
    for key,value in dict_obj.items():
        if isinstance(value,dict):
            sub_class = cfg.__dict__[key] if key in cfg.__dict__ else cfg.__class__.__dict__[key]
            if isinstance(sub_class,type):
                config_fill_in_from_dict(sub_class,value)
            else:
                setattr(cfg,key,value)
        else:
            setattr(cfg,key,value)
    return cfg

def read_custom_cfg(original_env_cfg,original_train_cfg,dir_path):
    import pickle
    with open(dir_path+"/env_config_dict.pkl","rb") as  file:
        env_cfg_dir=pickle.load(file)
    with open(dir_path+"/train_config_dict.pkl","rb") as  file:
        train_cfg_dir=pickle.load(file)
    print("Reading custom cfg from ",dir_path)
    print("*"*20)
    env_cfg=config_fill_in_from_dict(original_env_cfg,env_cfg_dir)
    train_cfg=config_fill_in_from_dict(original_train_cfg,train_cfg_dir)
    return env_cfg,train_cfg

def config_print_roll_out(class_name,space_num,log_file_name):
    space_num_added=False
    SPACE=4
    if space_num is not 0:
        log_file_name.write(" "*space_num*SPACE+ '|- '+"class "+class_name.__name__+"\n")
    for attr in dir(class_name):
        if  not attr.startswith('__'):
            if isinstance(getattr(class_name, attr),type):

                if space_num_added is False:
                    space_num+=1
                    space_num_added=True

                config_print_roll_out(getattr(class_name, attr),space_num,log_file_name)
            else:
                log_file_name.write(" "*(space_num+1)*SPACE+ '|- '+str(attr) + '=' +str( getattr(class_name, attr))+"\n")

def save_cfg(env_cfg, train_cfg,log_dir):
    os.makedirs(log_dir,exist_ok=True)
    env_log_file_name = log_dir + "/env_config"
    train_log_file_name= log_dir + "/train_config"
    import pickle
    # readable txt file
    with open(env_log_file_name+".txt","w") as  file:
        file.write(env_cfg.__class__.__name__+"\n")
        config_print_roll_out(env_cfg,0,file)

    with open(train_log_file_name+".txt","w") as  file:
        file.write(train_cfg.__class__.__name__+"\n")
        config_print_roll_out(train_cfg,0,file)

    # packed OBJ file of dict
    with open(env_log_file_name+"_dict.pkl","wb") as  file:
        env_cfg_dict={}
        config_roll_out2dict(env_cfg,env_cfg_dict)
        pickle.dump(env_cfg_dict,file)

    with open(train_log_file_name+"_dict.pkl","wb") as  file:
        traincfg_dict={}
        config_roll_out2dict(train_cfg,traincfg_dict)
        pickle.dump(traincfg_dict,file)

class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
