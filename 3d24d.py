import numpy as np
import matplotlib.pyplot as plt
import cv2 
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='Play and record from cameras.')
    parser.add_argument('--left_video', type=str, help='left_video_path')
    parser.add_argument('--right_video', type=str, help='right_video_path')
    parser.add_argument('--eef_pose', type=str, help='eef_pose_path')
    parser.add_argument('--camera_params_3d24d', type=str, help='camera_params_path')
    parser.add_argument('--camera_params_4d23d', type=str, default=None, help='camera_params_path')
    return parser.parse_args()


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
    return np.array(frames)

def get_camera_params(left_image_coordinates,right_image_coordinates,eef_pose):
    assert len(left_image_coordinates) == len(right_image_coordinates)
    assert len(left_image_coordinates) == len(eef_pose)
    Y = np.concatenate([left_image_coordinates,right_image_coordinates],axis = -1)
    X = np.concatenate([eef_pose,np.ones((len(eef_pose),1))],axis = -1)
    W_3d24d = np.linalg.pinv(X)@Y
    X = np.concatenate([left_image_coordinates,right_image_coordinates,np.ones((len(left_image_coordinates),1))],axis = -1)
    Y = eef_pose
    W_4d23d = np.linalg.pinv(X)@Y
    return W_3d24d,W_4d23d
    # return np.linalg.pinv(X.T@X+0.01*np.eye(4))@X.T@Y

class Callibrate():
    def __init__(self,args):
        self.left_images = read_video(args.left_video)
        self.right_images = read_video(args.right_video)
        self.eef_pose = np.load(args.eef_pose,allow_pickle=True)
        self.m = self.left_images.shape[1]
        self.n = self.left_images.shape[2]
        print("shape:",self.m,",",self.n)


        if args.camera_params_4d23d is None or args.camera_params_3d24d is None:
            self.indices = np.arange(self.left_images.shape[0])
            np.random.shuffle(self.indices)

            self.left_image_coordinates = []
            self.right_image_coordinates = []
            self.final_indices = []
            count = 0
            for i in self.indices:
                self.get_marker(self.left_images[i])
                if self.x is None or self.y is None:
                    print("skipping")
                    continue
                self.left_image_coordinates.append((self.x,self.y))
                self.get_marker(self.right_images[i])
                self.right_image_coordinates.append((self.x,self.y))
                count += 1
                self.final_indices.append(i)
                print("count:",count)
                if count >= 10:
                    break
            self.left_image_coordinates = np.array(self.left_image_coordinates)
            self.right_image_coordinates = np.array(self.right_image_coordinates)
            self.final_indices = np.array(self.final_indices)
            self.W_3d24d,self.W_4d23d = get_camera_params(self.left_image_coordinates,self.right_image_coordinates,self.eef_pose[self.final_indices])

            save_path_3d24d,save_path_4d23d = get_camera_params_save_path(args.eef_pose)

            np.save(save_path_3d24d,self.W_3d24d)
            print("camera params  3d24d saved to:",save_path_3d24d)
            np.save(save_path_4d23d,self.W_4d23d)
            print("camera params 4d23d saved to:",save_path_4d23d)

        else:
            assert args.camera_params_3d24d is not None and args.camera_params_4d23d is not None
            self.W_3d24d = np.load(args.camera_params_3d24d,allow_pickle=True)
            self.W_4d23d = np.load(args.camera_params_4d23d,allow_pickle=True)

    def get_marker(self,image):
        fig, ax =  plt.subplots()
        ax.imshow(image)
        fig.canvas.mpl_connect('button_press_event', self.press) 
        plt.show()

    def press(self,event):
        self.x,self.y  = event.xdata,event.ydata
        if self.x is not None and self.y is not None:
            self.x = self.x/self.n
            self.y = self.y/self.m
        print("Press",self.x,self.y)

def get_camera_params_save_path(eef_pose_path):
    assert os.path.exists(eef_pose_path) and eef_pose_path.endswith('.npy')
    parent = "/".join(eef_pose_path.split("/")[:-1])
    camera_params_path_3d24d = os.path.join(parent,"camera_params_3d24d.npy")
    camera_params_path_4d23d = os.path.join(parent,"camera_params_4d23d.npy")
    return camera_params_path_3d24d,camera_params_path_4d23d

def get_tmp_video_paths(eef_pose_path):
    assert os.path.exists(eef_pose_path) and eef_pose_path.endswith('.npy')
    parent = "/".join(eef_pose_path.split("/")[:-3])

    left_video_path = os.path.join(parent,"visualization","tmp_left_video.mp4")
    right_video_path = os.path.join(parent,"visualization","tmp_right_video.mp4")
    return left_video_path,right_video_path

def write_video(frames,coordinates,video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
    for i in range(len(frames)):
        frame = frames[i].copy()
        x,y = coordinates[i]
        cv2.circle(frame,(int(x),int(y)),1,(255,0,0),-1)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

def main(args):
    c = Callibrate(args)
    circle_coordinates = np.concatenate([c.eef_pose,np.ones((c.eef_pose.shape[0],1))],axis=-1)@c.W_3d24d
    circle_coordinates = circle_coordinates*c.m
    left_image_coordinates,right_image_coordinates = circle_coordinates[:,:2],circle_coordinates[:,2:4]
    left_video_path,right_video_path = get_tmp_video_paths(args.eef_pose)
    write_video(c.left_images,left_image_coordinates,left_video_path)
    write_video(c.right_images,right_image_coordinates,right_video_path)
    print("left video saved to:",left_video_path)
    print("right video saved to:",right_video_path)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)