import argparse
import hebi
import time
from time import sleep 
# from matplotlib import pyplot as plt
import numpy as np


from scipy.spatial.transform import Rotation 
# import transforms3d as t3d

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion,TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header,String,Float32MultiArray
from std_srvs.srv import Trigger
from franka_msgs.action import Move,Grasp



from tf2_ros import TransformBroadcaster

from cv2_enumerate_cameras import enumerate_cameras
import cv2
import os

CMD_RATE = 20 #hz




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
            print(f"Camera {idx} is available.")
            caps.append((idx,cap))
        else:
            print(f"Camera {idx} is not available.")
    return caps



def destroy_all_windows(idx_caps):
    for idx,cap in idx_caps:
        cap.release()
    cv2.destroyAllWindows()

def write_video(images,output_file_name):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file_name+".mp4", fourcc, 30.0, (120, 120))

    for j in range(images.shape[0]):
        img = images[j].copy()
        img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_AREA)
        video.write(img)

    video.release()


class Teleop(Node):

    def __init__(self,group,fbk,args):
        super().__init__('teleop')
        self.args = args
        idx_caps = get_idx_caps("Brio")
        print(f"Number of cameras: {len(idx_caps)}")
        if len(idx_caps) ==  0:
            print("Not enough cameras found.")
            exit(1)

        self.idx_caps = idx_caps
        self.frames = {}
        self.frames["robot0_eef_pos"] = []
        self.frames["robot0_eef_quat"] = []
        self.frames["actions"] = []

        self.frame = {}
        self.frame["robot0_eef_pos"] = None
        self.frame["robot0_eef_quat"] = None
        self.frame["actions"] = None
        for idx,cap in self.idx_caps:
            self.frames[idx] = []
            self.frame[idx] = None

        ret = False
        while not ret:
            ret = self.read_cameras()

        #fbk init
        self.group = group
        self.fbk = fbk
        self.get_feeback()
        while self.fbk is None:
            self.get_feeback()

        #readio init
        self.buttons = [0]*8
        self.sliders = [0]*8
        self.OPEN = True 
        self.CLOSE = not self.OPEN


        self.publisher = self.create_publisher(Float32MultiArray,"/input2action_command",10)
        self.subscription_current_pose = self.create_subscription(PoseStamped,'/franka_robot_state_broadcaster/current_pose',self.pose_subscription_callback,10)

        self.timer = self.create_timer(1/CMD_RATE, self.timer_callback,)
        self.camera_timer = self.create_timer(1/30, self.read_cameras)

        sleep(2)
        print("started")
    
        self.tick = time.time()

    def append_frame(self,frame):
        for idx,cap in self.idx_caps:
                self.frames[idx].append(frame[idx])
        self.frames["actions"].append(frame["actions"])
        self.frames["robot0_eef_pos"].append(frame["robot0_eef_pos"])
        self.frames["robot0_eef_quat"].append(frame["robot0_eef_quat"])


    def read_cameras(self):
        ret = True
        prev_frame = self.frame.copy()
        for idx,cap in self.idx_caps:
            r, f = cap.read()
            ret = ret and r
            self.frame[idx] = f
        if not ret:
            self.frame = prev_frame
        return ret


    def pose_subscription_callback(self,msg):
        self.frame["robot0_eef_pos"] = np.array([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z])
        self.frame["robot0_eef_quat"] = np.array([msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w])

    


    def timer_callback(self):

        self.read_io()
        if self.buttons[0] == 1:
            self.OPEN = True
            self.CLOSE = False
        if self.buttons[1] ==1:
            self.CLOSE = True
            self.OPEN = False

        self.frame["actions"] = self.sliders[:6]+[0.0]
        if self.OPEN:self.frame["actions"][6] = -1.0
        else:self.frame["actions"][6] = 1.0
        if (self.frame["robot0_eef_pos"] is None) or (self.frame["robot0_eef_quat"] is None):
            return
        self.append_frame(self.frame)

        if self.buttons[2]==1:
            tock = time.time()
            print("time_taken:",tock-self.tick)
            print("fps:",len(self.frames["actions"])/(tock-self.tick))
            os.makedirs("data/mp4/pick_and_place", exist_ok=True) 
            for idx,cap in self.idx_caps:
                write_video(np.array(self.frames[idx]),f"data/mp4/pick_and_place/robot_demo_{self.args.demo_idx}_camidx_{idx}")
                # np.save(f"data/mp4/pick_and_place/robot_demo_{self.args.demo_idx}_camidx_{idx}.npy",np.array(self.frames[idx]))
            np.save(f"data/mp4/pick_and_place/robot_demo_{self.args.demo_idx}_actions.npy",np.array(self.frames["actions"]))
            np.save(f"data/mp4/pick_and_place/robot_demo_{self.args.demo_idx}_robot0_eef_pos.npy",np.stack(self.frames["robot0_eef_pos"]))
            np.save(f"data/mp4/pick_and_place/robot_demo_{self.args.demo_idx}_robot0_eef_quat.npy",np.stack(self.frames["robot0_eef_quat"]))
            print("done")
            exit()

        msg = Float32MultiArray()
        msg.data = self.frame["actions"]
        self.publisher.publish(msg)
        


    def get_feeback(self):
        prev_fbk = self.fbk
        fbk = self.group.get_next_feedback(reuse_fbk=self.fbk)
        if fbk is None:
            self.fbk = prev_fbk
        else:
            self.fbk = fbk
        return self.fbk
    
    def read_io(self):
        self.get_feeback()
        for i in range(8):
            self.buttons[i] = self.fbk.io.b.get_int(i + 1).item()
        for i in range(8):
            self.sliders[i] = self.fbk.io.a.get_float(i + 1).item()
        


def parse_arguments():
    parser = argparse.ArgumentParser(description='Play and record from cameras.')
    parser.add_argument('--demo_idx', type=int, default=0, help='Demo index')
    return parser.parse_args()

def main():

    args = parse_arguments()
    lookup = hebi.Lookup()

    # Wait 2 seconds for the module list to populate
    sleep(2.0)

    family_name = "HEBI"
    module_name = "mobileIO"

    group = lookup.get_group_from_names([family_name], [module_name])

    if group is None:
        print('Group not found: Did you forget to set the module family and name above?')
        exit(1)

    fbk = hebi.GroupFeedback(group.size)

    # print("GREAT")
    # exit(1)

    rclpy.init()
    teleop = Teleop(group,fbk,args)
    rclpy.spin(teleop)
    teleop.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
