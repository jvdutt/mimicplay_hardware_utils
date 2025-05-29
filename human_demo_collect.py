import argparse

import time
from time import sleep 
import rclpy
from rclpy.node import Node
import numpy as np

from cv2_enumerate_cameras import enumerate_cameras
import cv2
import os

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

def write_video(images,output_file_name):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file_name+".mp4", fourcc, 30.0, (120, 120))

    for j in range(images.shape[0]):
        img = images[j].copy()
        img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_AREA)
        video.write(img)

    video.release()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Play and record from cameras.')
    parser.add_argument('--demo_idx', type=int, default=0, help='Demo index')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    return parser.parse_args()

class CameraNode(Node):
    def __init__(self,args):
        super().__init__('camera_node')
        self.args = args
        idx_caps = get_idx_caps("Brio")
        print(f"Number of cameras: {len(idx_caps)}")
        if len(idx_caps) ==  0:
            print("Not enough cameras found.")
            exit(1)

        self.frames = {}
        self.frame = {}
        self.idx_caps = idx_caps
        for idx,cap in self.idx_caps:
            self.frame[idx] = None
            self.frames[idx] = []


        ret = False
        while not ret:
            ret = self.read_cameras()

        print("started")
        self.tick = time.time()
        #timer1 = self.create_timer(0.01, self.read_cameras)
        timer2 = self.create_timer(1/self.args.fps, self.read_cameras)

    def read_cameras(self):
        ret = True
        prev_frame = self.frame.copy()
        for idx,cap in self.idx_caps:
            r, f = cap.read()
            ret = ret and r
            self.frame[idx] = f
        if not ret:
            self.frame = prev_frame
        self.append_frame()
        
        return ret

    def append_frame(self):
        for idx,cap in self.idx_caps:
                self.frames[idx].append(self.frame[idx])

def main():

    args = parse_arguments()
    print("FPS;",args.fps)
    try:
        rclpy.init()
        camera_node = CameraNode(args)
        rclpy.spin(camera_node)
        camera_node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        tock = time.time()
        print("time_taken:",tock-camera_node.tick)
        for idx in camera_node.frames:
            print("fps:",len(camera_node.frames[idx])/(tock-camera_node.tick))
            break
        c = input("want to save?[y/n]")
        if c.lower() == 'y':
            for idx in camera_node.frames:
                print("saving",idx)
                images = camera_node.frames[idx]
                images = [cv2.resize(img, (120, 120), interpolation=cv2.INTER_AREA) for img in images]
                images = np.array(images)
                print(images.shape)
                write_video(images,f"data/mp4/pick_and_place/human_demo_{args.demo_idx}_camidx_{idx}")
        else:
            print("not saving")
if __name__ == '__main__':
    main()
