import sys
l = ['robomimic',
     'MimicPlay',]
sys.path = l + sys.path
sys.path

import argparse
import json
import numpy as np
import time
import os
import psutil
import sys
import traceback
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.log_utils import PrintLogger, DataLogger

from mimicplay.configs import config_factory
from mimicplay.algo import algo_factory, RolloutPolicy
from mimicplay.utils.train_utils import get_exp_dir, rollout_with_stats, load_data_for_training

import mimicplay.utils.file_utils as FileUtils
import cv2 
import h5py
from functools import partial

from mimicplay.algo.GPT import *
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import torch.distributions as D 
import torchvision.transforms.functional as TF


# highlevel_ckpt_path = "/home/smart/Vishnu/docker_mimicplay_project/MimicPlay/trained_models_highlevel/robot_only_pick_and_place/20250514125510/models/model_epoch_682_best_validation_-76.04536361694336.pth"
# lowlevel_ckpt_path = "/home/smart/Vishnu/docker_mimicplay_project/MimicPlay/trained_models_lowlevel/robot_only_pick_and_place/20250516123256/models/model_epoch_100_tl_-12.87_vl_-12.13.pth"

from cv2_enumerate_cameras import enumerate_cameras
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header,String,Float32MultiArray

import warnings



def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Low Level Model")
    parser.add_argument("--config", type=str, 
                        default="/home/user/mimicplay_project/MimicPlay/mimicplay/configs/lowlevel.json", 
                        help="Path to the config file.")
    parser.add_argument("--highlevel_ckpt", type=str, 
                        # default="/home/user/mimicplay_project//MimicPlay/trained_models_highlevel/human_highlevel_epoch_200_best_validation_-75.72975082397461.pth",
                        default= "/home/user/mimicplay_project/MimicPlay/trained_models_highlevel/robot_highlevel_epoch_682_best_validation_-76.04536361694336.pth", 
                        help="Path to the high level checkpoint.")
    parser.add_argument("--lowlevel_ckpt", type=str, 
                        # default="/home/user/mimicplay_project/MimicPlay/trained_models_lowlevel/human_robot_lowlevel_tl_-34.27_vl_-34.30_best.pth",
                        default = "/home/user/mimicplay_project/MimicPlay/trained_models_lowlevel/robot_only_lowlevel_tl_-32.95_vl_-33.37_best.pth",
                        help="Path to the low level checkpoint.")
    parser.add_argument("--video_prompt", type=str,
                        # default="/home/user/mimicplay_project/MimicPlay/mimicplay/scripts/human_playdata_process/hand_object_detector/data/hdf5/pick_and_place/human_test.hdf5", 
                        default="/home/user/mimicplay_project/MimicPlay/mimicplay/scripts/human_playdata_process/hand_object_detector/data/hdf5/pick_and_place/robot_test.hdf5",
                        help="Path to the video prompt.")
    parser.add_argument("--dataset", type=str, 
                        # default="/home/user/mimicplay_project/MimicPlay/mimicplay/scripts/human_playdata_process/hand_object_detector/data/hdf5/pick_and_place/human_test.hdf5",
                        default="/home/user/mimicplay_project/MimicPlay/mimicplay/scripts/human_playdata_process/hand_object_detector/data/hdf5/pick_and_place/robot_test.hdf5",
                          help="Path to the dataset.")
    parser.add_argument("--right", action="store_true")
    parser.add_argument("--human",action="store_true",help="use human data")
    args = parser.parse_args()
    return args


def get_config_device(args):
    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = args.dataset

    # if args.name is not None:
    #     config.experiment.name = args.name

    if args.highlevel_ckpt is not None:
       config.algo.lowlevel.trained_highlevel_planner = args.highlevel_ckpt

    config.experiment.rollout.enabled = False
    config.train.num_epochs = 1000

    

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    config.lock()
    return config,device

class LowLevelModel(nn.Module):
    def __init__(self,config):
        super(LowLevelModel, self).__init__()
        self.gpt = self.get_gpt_model(config)
        self.image_enc = torch.nn.Sequential(*(list(models.resnet18().children())[:-2]),SpatialSoftmax(512, 3, 3, config.algo.lowlevel.spatial_softmax_num_kp))
        self.dropout_layer = nn.Dropout(config.algo.lowlevel.dropout)
        self.pose_enc = self.get_pose_encoder(config)
        self.pose_enc.float()
        self.decoder_mean,self.decoder_scale,self.decoder_logits = self.get_gmm_decoders(config)

        self.gmm_min_std = config.algo.lowlevel.gmm_min_std
        self.gmm_modes = config.algo.lowlevel.gmm_modes
        
    def forward(self,obs,eval=False):

        batch = obs
        b,t,c,h,w = batch["agentview_image"].shape
        model = self
        

        imgs = batch["robot0_eye_in_hand_image"].reshape(b*t,c,h,w).contiguous()
        img_enc = model.image_enc(imgs)
        img_enc = img_enc.reshape(b,t,-1).contiguous()

        x_pose = torch.cat((batch["robot0_eef_pos"], batch["robot0_eef_quat"]), dim=-1).float().contiguous()
        x_pose_feat = model.pose_enc(x_pose)

        latent_plan = batch["latent_plan"]
        input_tensor = torch.cat((latent_plan,img_enc, x_pose_feat), dim=-1).contiguous()
        output_tensor,_ = model.gpt(input_tensor)

        means = model.decoder_mean(output_tensor)
        means = torch.tanh(means)
        means = means.reshape(b,t,model.gmm_modes,-1).contiguous()


        scales = model.decoder_scale(output_tensor)
        scales = F.softplus(scales) + model.gmm_min_std
        scales = scales.reshape(b,t,model.gmm_modes,-1).contiguous()

        if eval:
            scales = torch.ones_like(scales).to(scales.device) * model.gmm_min_std

        logits = model.decoder_logits(output_tensor)

        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)  # shift action dim to event shape



        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        dists = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        return dists
    
    def save_model(self,save_dir,info):
        """
        Save the model to the specified directory.
        """
        os.makedirs(save_dir,exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, f'model_{info}.pth'))

    def load_model(self,ckpt_path):
        state_dict = torch.load(ckpt_path)
        self.load_state_dict(state_dict)
 



    def get_gpt_model(self,config):
        feat_dim = config.algo.lowlevel.feat_dim
        n_layer = config.algo.lowlevel.n_layer
        n_head = config.algo.lowlevel.n_head
        block_size = config.algo.lowlevel.block_size
        model_config = GPT.get_default_config()
        model_config.vocab_size = feat_dim
        model_config.n_embd = feat_dim
        model_config.n_layer = n_layer
        model_config.n_head = n_head
        model_config.block_size = block_size
        return GPT(model_config)
    
    def get_pose_encoder(self,config):
        return  nn.Sequential(
                           nn.Linear(config.algo.lowlevel.proprio_dim, 32),
                           nn.ReLU(),
                           nn.Linear(32, 64),
                           nn.ReLU(),
                           nn.Linear(64, 128),
                          )
    
    def get_gmm_decoders(self,config):
        feat_dim = config.algo.lowlevel.feat_dim
        gmm_modes = config.algo.lowlevel.gmm_modes
        action_dim = config.algo.lowlevel.action_dim

        mlp_decoder_mean = nn.Sequential(
                           nn.Linear(feat_dim, 256),
                           nn.ReLU(),
                           nn.Linear(256, 128),
                           nn.ReLU(),
                           nn.Linear(128, gmm_modes * action_dim)
                          )

        mlp_decoder_scale = nn.Sequential(
                           nn.Linear(feat_dim, 256),
                           nn.ReLU(),
                           nn.Linear(256, 128),
                           nn.ReLU(),
                           nn.Linear(128, gmm_modes * action_dim)
                          )

        mlp_decoder_logits = nn.Sequential(
                           nn.Linear(feat_dim, 256),
                           nn.ReLU(),
                           nn.Linear(256, 64),
                           nn.ReLU(),
                           nn.Linear(64, gmm_modes)
                          )
        return mlp_decoder_mean,mlp_decoder_scale,mlp_decoder_logits
    

def policy_from_checkpoint(device=None, ckpt_path=None, ckpt_dict=None, verbose=False, update_obs_dict=True):
    """
    This function restores a trained policy from a checkpoint file or
    loaded model dictionary.

    Args:
        device (torch.device): if provided, put model on this device

        ckpt_path (str): Path to checkpoint file. Only needed if not providing @ckpt_dict.

        ckpt_dict(dict): Loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.

        verbose (bool): if True, include print statements

    Returns:
        model (RolloutPolicy): instance of Algo that has the saved weights from
            the checkpoint file, and also acts as a policy that can easily
            interact with an environment in a training loop

        ckpt_dict (dict): loaded checkpoint dictionary (convenient to avoid
            re-loading checkpoint from disk multiple times)
    """
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path, ckpt_dict=ckpt_dict)

    # algo name and config from model dict
    algo_name, _ = FileUtils.algo_name_from_checkpoint(ckpt_dict=ckpt_dict)

    config, _ = FileUtils.config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict, verbose=verbose)

    if update_obs_dict:
        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        ObsUtils.initialize_obs_utils_with_config(config)

    # env meta from model dict to get info needed to create model
    env_meta = ckpt_dict["env_metadata"]
    shape_meta = ckpt_dict["shape_metadata"]

    # maybe restore observation normalization stats
    obs_normalization_stats = ckpt_dict.get("obs_normalization_stats", None)
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        for m in obs_normalization_stats:
            for k in obs_normalization_stats[m]:
                obs_normalization_stats[m][k] = np.array(obs_normalization_stats[m][k])

    if device is None:
        # get torch device
        device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # create model and load weights
    model = algo_factory(
        algo_name,
        config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    # print(ckpt_dict["model"])
    model.deserialize(ckpt_dict["model"])
    model.set_eval()
    # model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)
    # if verbose:
    #     print("============= Loaded Policy =============")
    #     print(model)
    return model, ckpt_dict




def get_projection(eef_pos,W): 
    return torch.concatenate([eef_pos,torch.ones((*eef_pos.shape[:-1],1)).to(eef_pos.device)],axis=-1) @ W


def _get_latent_plan(self, obs, goal):
    assert 'agentview_image' in obs.keys() # only visual inputs can generate latent plans

    if len(obs['agentview_image'].size()) == 5:
        bs, seq, c, h, w = obs['agentview_image'].size()

        for item in ['agentview_image']:
            obs[item] = obs[item].view(bs * seq, c, h, w)
            goal[item] = goal[item].view(bs * seq, c, h, w)
            
        tmp = obs['robot0_eef_pos'].view(bs * seq, 3)
      
        obs['robot0_eef_pos'] = get_projection(tmp,W_3d24d)

        dists, enc_out, mlp_out = self.nets["policy"].forward_train(
            obs_dict=obs,
            goal_dict=goal,
            return_latent=True
        )

        obs['robot0_eef_pos'] = tmp

        act_out_all = dists.mean
        act_out = act_out_all

        for item in ['agentview_image']:
            obs[item] = obs[item].view(bs, seq, c, h, w)
            goal[item] = goal[item].view(bs, seq, c, h, w)

        obs['robot0_eef_pos'] = obs['robot0_eef_pos'].view(bs, seq, 3)

        enc_out_feature_size = enc_out.size()[1]
        mlp_out_feature_size = mlp_out.size()[1]

        mlp_out = mlp_out.view(bs, seq, mlp_out_feature_size)
    else:
        dists, enc_out, mlp_out = self.nets["policy"].forward_train(
            obs_dict=obs,
            goal_dict=goal,
            return_latent=True
        )

        act_out_all = dists.mean
        act_out = act_out_all

    return act_out, mlp_out




def get_model(config, device,lowlevel_ckpt_path=None, highlevel_ckpt_path=None):
    """
    Create a model from the config and device.
    """

    model = LowLevelModel(config)

    ckpt_path = lowlevel_ckpt_path
    # ckpt_path = None
    if ckpt_path is not None:
        # load model weights from checkpoint
        model.load_model(ckpt_path)
        print("Loaded model from checkpoint: {}".format(ckpt_path))
    #load or initialize model weights
    model = model.to(device)
    #human_net 
    human_nets, _ = policy_from_checkpoint(ckpt_path=highlevel_ckpt_path, device=device, verbose=False,update_obs_dict=False)
    model.human_nets = human_nets
    if args.human:
        model.human_nets._get_latent_plan = partial(_get_latent_plan,model.human_nets)
    model.device = device
    model.eval()
    return model


def get_idx_caps(key_name=""):
    indices = []
    caps = []
    for camera_info in enumerate_cameras(cv2.CAP_GSTREAMER):
        if key_name in camera_info.name:
            indices.append(camera_info.index)
            print(f'{camera_info.index}: {camera_info.name}')

    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 120)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
            print(f"Camera {idx} is available.")
            caps.append((idx,cap))
        else:
            print(f"Camera {idx} is not available.")
    return caps

def destroy_all_windows(idx_caps):
    for idx,cap in idx_caps:
        cap.release()
    cv2.destroyAllWindows()

def assign_cameras_indices(idx_caps):

    frame = {}
    for idx,cap in idx_caps:
        frame[idx] = None

    prev_time = 0.0
    while True:
        ret = True
        for idx,cap in idx_caps:
            r,f = cap.read()
            ret = ret and r
            frame[idx] = f

        if ret:
            for idx,cap in idx_caps:
                #frames[idx].append(frame[idx])
                cv2.imshow(f'Camera {idx}', frame[idx])
        else:
            print(f"Failed to read from camera ")
        c = cv2.waitKey(1)
        if c == 27:
            cv2.destroyAllWindows()
            break
    s = input("order of the indices => (left,right,wrist):")
    s = [int(_) for _ in s.split()]
    assert len(s) == 3
    return s


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    images = np.array(frames)
    images = np.array([cv2.resize(img, (84, 84)) for img in images])
    return images


class MimicPlay_Control(Node):

    def __init__(self,model,device,is_right,video_prompt_path):
        super().__init__('mimicplay_control')
        self.model = model
        self.device = device
        self.is_right = is_right
     
        self.load_eval_video_prompt(video_prompt_path)
        self.current_id = 0
        self.zero_count = 0

                
        self.idx_caps = get_idx_caps("Brio")
        print(f"Number of cameras: {len(self.idx_caps)}")
        if len(self.idx_caps) ==  0:
            print("Not enough cameras found.")
            exit(1)

        # print(len(self.goal_image))
        # print(self.goal_image.shape)

        # left_idx,right_idx,wrist_idx = assign_cameras_indices(self.idx_caps)
        left_idx, right_idx, wrist_idx = 4,6,2
        if is_right:
            agentview_idx = right_idx
        else:
            agentview_idx = left_idx
        self.agentview_idx = agentview_idx
        self.wrist_idx = wrist_idx
        

        self.max_buffer_size = 10
        self.buffer= {"obs":{},"goal_obs":{}}
        self.buffer["obs"]["robot0_eef_pos"] = []
        self.buffer["obs"]["robot0_eef_quat"] = []
        self.buffer["obs"]["agentview_image"] = []
        self.buffer["obs"]["robot0_eye_in_hand_image"] = []
        self.buffer["goal_obs"]["agentview_image"] = []

        self.frame = {}
        self.frame["robot0_eef_pos"] = None
        self.frame["robot0_eef_quat"] = None
        self.frame["agentview_image"] = None
        self.frame["robot0_eye_in_hand_image"] = None


        #reading images at 30 fps
        ret = False
        while not ret:
            ret = self.read_cameras()
        self.camera_timer = self.create_timer(1/30, self.read_cameras)

        input("waiting for input")
        #eef pose,quat 
        self.subscription_current_pose = self.create_subscription(PoseStamped,'/franka_robot_state_broadcaster/current_pose',self.pose_subscription_callback,10)


        #final control
        self.eval_goal_img_window = 30
        self.eval_max_goal_img_iter = 10
        self.eval_goal_gap = 150
        if args.human:
            self.eval_goal_gap = 50
            self.eval_goal_img_window = 10
        self.control_timer = self.create_timer(1/20,self.control_callback)
        self.publisher = self.create_publisher(Float32MultiArray,"/input2action_command",10)

        self.show_images = self.create_timer(1/30,self.show_images_callback)

        # input()
        print("started")
 



    def control_callback(self):
        obs = self.frame.copy()
        ee_pos = obs["robot0_eef_pos"]

        if not hasattr(self,"prev_gripper"):
            self.prev_gripper = []
        if not hasattr(self,"prev_actions"):
            self.prev_actions = []
       
        print(self.current_id)

        self.goal_id = min(self.current_id + self.eval_goal_gap, self.goal_image_length - 1)
        goal_obs = {'agentview_image': self.goal_image[self.goal_id:(self.goal_id+1)]}

        obs,goal_obs = self.append_buffer(obs.copy(),goal_obs.copy())
   

        actions,actions_mean = self.get_actions(obs,goal_obs)

        actions = actions[0][-1]
        actions_mean = actions_mean[0][-1]
        actions = np.clip(actions, -1, 1)
        actions_mean = np.clip(actions_mean, -1, 1)

        
        # self.prev_gripper.append(actions[-1])
        # if len(self.prev_gripper) > 10:
        #     self.prev_gripper.pop(0)
        # self.prev_actions.append(actions)
        # if len(self.prev_actions) > 3:
        #     self.prev_actions.pop(0)

        # actions = np.mean(np.array(self.prev_actions),axis=0)
        # actions[-1] = np.mean(self.prev_gripper)

        actions[-1] = actions_mean[-1]
        # actions = actions_mean
        print(actions)
        # print(actions)
        ee_pos = ee_pos.to(self.device)
        if args.human:
            ee_pos = get_projection(ee_pos.reshape(1,3),W_3d24d).reshape(-1)
            ee_pos[0],ee_pos[1] = ee_pos[1],ee_pos[0]
            ee_pos[2],ee_pos[3] = ee_pos[3],ee_pos[2]
        self.current_id = self.find_nearest_index(ee_pos, self.current_id)

        self.publisher.publish(Float32MultiArray(data=actions))


    def find_nearest_index(self, ee_pos, current_id):
        min_i = max(0, current_id - self.eval_goal_img_window)
        max_i = min(current_id + self.eval_goal_img_window, self.goal_image_length - 1)

        distances = torch.norm(self.goal_ee_traj[min_i:max_i] - ee_pos, dim=1)
        nearest_index = distances.argmin().item()
        if nearest_index == 0:
            self.zero_count += 1
        # if self.zero_count > self.eval_max_goal_img_iter:
        #     nearest_index += 1
        #     self.zero_count = 0

        return min(nearest_index + current_id, self.goal_image_length - 1)


    def get_actions(self,obs,goal_obs):
        obs["agentview_image"] = obs["agentview_image"].to(self.device).float().permute(0,1,4,2,3)/(255.0)
        obs["robot0_eye_in_hand_image"] = obs["robot0_eye_in_hand_image"].to(self.device).float().permute(0,1,4,2,3)/(255.0)
        obs["robot0_eef_pos"] = obs["robot0_eef_pos"].to(self.device).float()
        obs["robot0_eef_quat"] = obs["robot0_eef_quat"].to(self.device).float() 

        with torch.no_grad():
            _, mlp_feature = self.model.human_nets._get_latent_plan(obs, goal_obs)
            obs['latent_plan'] = mlp_feature.detach()
            dists = self.model(obs)
        return dists.sample().detach().cpu().numpy(),dists.mean.detach().cpu().numpy()
    
    def append_list(self,lis,val):
        lis.append(val)
        if len(lis) > self.max_buffer_size:
            lis.pop(0)
        return lis
    
    def append_buffer(self,obs,goal_obs):

        for key in obs.keys():
            if key in self.buffer["obs"]:
                self.buffer["obs"][key] = self.append_list(self.buffer["obs"][key],obs[key])
                obs[key] = torch.stack(self.buffer["obs"][key],dim=0)[None]
        self.buffer["goal_obs"]["agentview_image"] = self.append_list(self.buffer["goal_obs"]["agentview_image"],goal_obs["agentview_image"])
        goal_obs["agentview_image"] = torch.stack(self.buffer["goal_obs"]["agentview_image"],dim=0)[None]
        return obs,goal_obs

    
    def show_images_callback(self):
        if self.frame["agentview_image"] is not None and self.frame["robot0_eye_in_hand_image"] is not None:
            cv2.imshow(f'a', self.frame["agentview_image"].numpy()[:,:,::-1])
            cv2.imshow(f'r', self.frame["robot0_eye_in_hand_image"].numpy()[:,:,::-1])
            cv2.imshow(f"g",self.goal_image[self.goal_id].detach().permute(1,2,0).cpu().numpy()[:,:,::-1])
            self.cv_key = cv2.waitKey(1)



    
    def load_eval_video_prompt(self, video_path):
        demo_idx = f"demo_{int(self.is_right)}"
        if args.human:
            demo_idx = f"demo_{1-int(self.is_right)}"
        with h5py.File(video_path, 'r') as f:
            self.goal_image = f['data'][demo_idx]['obs']['agentview_image'][:]
            self.goal_ee_traj = f['data'][demo_idx]['obs']['robot0_eef_pos'][:]
        self.goal_image = torch.from_numpy(self.goal_image).to(self.device).float()
        self.goal_ee_traj = torch.from_numpy(self.goal_ee_traj).to(self.device).float()
        self.goal_image = self.goal_image.permute(0, 3, 1, 2)
        self.goal_image = self.goal_image / 255.
        self.goal_image_length = len(self.goal_image)
        self.goal_id = 150

    def read_cameras(self):
        ret = True
        prev_frame = self.frame.copy()
        ret = True
        for idx,cap in self.idx_caps:
            r, f = cap.read()
            if r:f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            ret = ret and r
            if idx == self.agentview_idx:
                self.frame["agentview_image"] = torch.from_numpy(cv2.resize(f, (84, 84), interpolation=cv2.INTER_AREA))
            if idx == self.wrist_idx:
                self.frame["robot0_eye_in_hand_image"] = torch.from_numpy(cv2.resize(f, (84, 84), interpolation=cv2.INTER_AREA))
        if not ret:
            self.frame = prev_frame
        return ret
    
    def pose_subscription_callback(self,msg):
            self.frame["robot0_eef_pos"] = torch.tensor([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z])
            self.frame["robot0_eef_quat"] = torch.tensor([msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w])



class attrDict(dict):
    """
    A dictionary that allows attribute-style access to its keys.
    """
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            return None

def recursive_dict(d):
    for k in d:
        if isinstance(d[k], dict):
            d[k] = recursive_dict(d[k])
    return attrDict(d)



if __name__ == "__main__":
    args = parse_args()


    # print("got here1")

 

    config,device = get_config_device(args)
    # device = torch.device("cpu")
    ObsUtils.initialize_obs_utils_with_config(config)

    lowlevel_ckpt_path = args.lowlevel_ckpt
    highlevel_ckpt_path = args.highlevel_ckpt


    if args.human:
        W_3d24d = np.load("/home/user/mimicplay_project//MimicPlay/camera_params_3d24d.npy")
        W_3d24d = torch.from_numpy(W_3d24d).to(device).float()

    # print("got here2")
    # print("---"*10)

    # device = torch.device("cpu")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = get_model(config, device,lowlevel_ckpt_path=lowlevel_ckpt_path, highlevel_ckpt_path=highlevel_ckpt_path)



    # state_dict = torch.load(ckpt_path,weights_only=True)
    # print("Loaded model from checkpoint: {}".format(ckpt_path))
    #load or initialize model weights
    # model = model.to(device)

    # print(device)

    rclpy.init()
    control = MimicPlay_Control(model,device,args.right,args.video_prompt)
    rclpy.spin(control)
    control.destroy_node()
    rclpy.shutdown()
    

