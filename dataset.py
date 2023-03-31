import os
import scipy.io as sio
import numpy as np
import cv2
import math
from utils.math import make_chunks
class Dataset(object):
    def __init__(self, video_path, feat_path, gt_path):
        self.video_path = video_path
        self.feat_path = feat_path
        self.gt_path = gt_path
    
    def split_train_test(self, test_name = []):
        self.test_name = test_name
        self.all_name = [file.split('_gt')[0] for file in os.listdir(self.gt_path)]
        self.train_name = [name for name in self.all_name if name not in self.test_name]
        self.train_name.sort()
        
    def get_video_name(self, video_index, is_trainset):
        if is_trainset:
            name = self.train_name[video_index]
        else:
            name = self.test_name[video_index]
        return name
        
    def get_video_feat(self, video_name):
        feat_file= sio.loadmat(self.feat_path+video_name+'_alex_fc7_feat.mat')
        feat = feat_file['Features'] #(n_frame, 1, 4096)
        feat = np.squeeze(feat, axis=(1,))
        return feat
    
    def get_video_gt(self, video_name):
        gt_file = sio.loadmat(self.gt_path+video_name+'_gt.mat')
        gt = gt_file['gt']  # (1, n_frame)
        return gt[0]
        
    def get_video_data(self, video_index, is_trainset):
        video_name = self.get_video_name(video_index, is_trainset)
        feat = self.get_video_feat(video_name)
        gt = self.get_video_gt(video_name)
        return video_name, feat, gt
    
    def get_video_capture(self, video_index, is_trainset):
        name = self.get_video_name(video_index, is_trainset)
        cap = cv2.VideoCapture(self.video_path + name + '.MP4')
        return name, cap
    
    def load_gt_segments(self):
        self.import_segment = {}
        raise NotImplementedError()
        
    
class TVSum(Dataset):
            
    def load_gt_segments(self):
        self.import_segment = {}
        for video_name in self.all_name:
            temp_2 = sio.loadmat(os.path.join(self.gt_path, video_name+'_gt.mat'))
            gt = temp_2['gt'][0]  # (1, n_frame) -> (n_frame)
            
            cap = cv2.VideoCapture(os.path.join(self.video_path, video_name+'.mp4'))
            fps = math.ceil(cap.get(5))
            
            segment_duration = 2
            segment_len = fps * segment_duration
            
            segments_gt = [(index*segment_len, index*segment_len + len(segment)-1) \
                        for index, segment in enumerate(make_chunks(gt, segment_len)) if segment[0]==1]
            self.import_segment[video_name] = segments_gt
    
    

class Tour20(Dataset):
    def load_gt_segments(self, segment_path):
        self.import_segment = {}
        for video_name in self.all_name:
            temp_2 = sio.loadmat(os.path.join(self.gt_path, video_name+'_gt.mat'))
            gt = temp_2['gt'][0]  # (1, n_frame) -> (n_frame)
            num_frames = len(gt)

            # get all segment, index frame start from 0
            file_path = os.path.join(segment_path, video_name[:2], 'frm_num_'+video_name)
            with open(file_path, 'r') as f:
                rows = [row[:-1] for row in f.readlines()]  # remove \n
                shots = [start_end for start_end in make_chunks(rows, 2)]
                for i in range(len(shots)):
                    shot = shots[i]
                    shots[i] = [int(shot[0]) - 1, int(shot[1]) - 1]
                # shots[-1][-1] = max(shots[-1][-1], num_frames-1)
                shots[-1][-1] = num_frames-1

            # filter true segment
            shots = [(start, end) for start, end in shots if gt[start+1]==1]
            self.import_segment[video_name] = shots