import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse



class Stabilize():

    def __init__(self,
                 input_path,
                 output_path = None,
                 n_key_pts_limit = 500,
                 refresh_interval = 100,
                 frame_ref = 0,
                 frame_ini = 1,
                 frame_fin = -1,
                 sampling  = 1,
                 zoom = 1.0,
                 nfeatures = 2000
                ):

        self.input_path = input_path
        cwd = os.getcwd()
        if output_path == None:
            file_name = os.path.splitext(input_path)[0]
            self.output_path = os.path.join(cwd, file_name+'_stb.avi')
            self.output_stack_path = os.path.join(cwd, file_name+'_stb_stack.avi')
        else:
            self.output_path = os.path.join(cwd, output_path)
            file_name = os.path.splitext(output_path)[0]
            self.output_stack_path = os.path.join(cwd, file_name+'_stb_stack.avi')

        self.cap = cv2.VideoCapture(self.input_path)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.n_key_pts_limit = n_key_pts_limit
        self.refresh_interval = refresh_interval
        self.frame_ref = frame_ref
        self.frame_ini = frame_ini
        self.sampling = sampling
        if frame_fin == -1:
            self.frame_fin = self.n_frames-1
        else:
            self.frame_fin = frame_fin

        self.nfeatures = nfeatures
        self.zoom = zoom

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #out_stack = cv2.VideoWriter(self.output_stack_path, fourcc, self.fps, (2*int(self.w), int(self.h)))
        out_stack = cv2.VideoWriter(self.output_stack_path, fourcc, self.fps, (self.w, self.h))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_ref)
        success, prev_img = self.cap.read()
        assert success == True, f"Problems reading file: {self.input_path}"
        print('fps, n_frames, w, h: ', self.fps, self.n_frames, self.w, self.h)

        prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create(nfeatures=self.nfeatures)
        prev_kp, prev_des = sift.detectAndCompute(prev_img_gray,mask=None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)


        m_prev = np.eye(3)
        ms = []
        n_matches = []
        refresh_list = []
        flag = False
        for i in tqdm(range(self.frame_ini,self.frame_fin,self.sampling)):

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, curr = self.cap.read()
            if not success:
                break

            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            curr_kp, curr_des = sift.detectAndCompute(curr_gray,mask=None)

            matches = flann.knnMatch(prev_des,curr_des,k=2)

            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)

            prev_pts = np.float32([ prev_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            curr_pts = np.float32([ curr_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            m, inlier = cv2.findHomography(curr_pts, prev_pts, cv2.RANSAC,5.0)

            m = np.matmul(m, m_prev)


            curr_stabilized = cv2.warpPerspective(curr, m, (self.w,self.h))

            curr_stabilized = fixBorder(curr_stabilized, zoom=self.zoom)

            #curr_out = cv2.hconcat([curr_stabilized])

            out_stack.write(curr_stabilized)

            good_len = len(good)
            n_matches.append(good_len)
            if (i%self.refresh_interval==0):
                flag = True

            if (good_len > self.n_key_pts_limit) & flag:
                refresh_list.append(i)
                m_prev = m
                prev_kp, prev_des = curr_kp[:], curr_des[:]
                flag = False


        self.cap.release()
        out_stack.release()


def fixBorder(frame, zoom=1.5):
    if zoom > 1.0:
        s = frame.shape
        T = cv2.getRotationMatrix2D((s[1]/1.8, s[0]/1.8), 0, zoom)
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame




def parse_function():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='input file name')
    parser.add_argument('--output_path', type=str, help='output file name (optional)')
    parser.add_argument('--refresh_interval', type=int, default=100, help='refresh interval (default value = 100)')
    parser.add_argument('--frame_ref', type=int, default = 0, help='reference frame use to anchor the video')
    parser.add_argument('--frame_ini', type=int, default = 1, help='initial frame')
    parser.add_argument('--frame_fin', type=int, default = -1, help='final frame')
    parser.add_argument('--sampling', type=int, default  = 1, help='sampling')
    parser.add_argument('--zoom', type=float, default = 1.0, help='zoom applied to the stabilized image for the stack output')
    parser.add_argument('--n_key_pts_limit', type=int, default  = 500, help='keypoint limit to accept refreshing.')
    parser.add_argument('--nfeatures', type=int, default  = 2000, help='Max number of keypoints')
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_function()

    stb = Stabilize(**vars(args))
    stb.run()
