import tensorflow as tf 
import numpy as np
from extract_feature_model import Extractor
from neural_network import NeuralNetwork
import scipy.io as sio
import cv2
from scipy.stats import norm

class Memo(object):
    def __init__(self):
        self.reset()
    
    def add(self, pre_info, state, target):
        self.pre_info_batch.append(pre_info)
        self.x_batch.append(state)
        self.y_batch.append(target)
    
    def get_size(self):
        return len(self.y_batch)
    
    def reset(self):
        self.pre_info_batch = []
        self.x_batch = []
        self.y_batch = []
    
class RewardEstimator(object):
    def __init__(self, window_size, sd_gauss):
        self.window_size = window_size
        self.sd_gauss = sd_gauss
        self.create_distribution
        
    def create_distribution(self):
        self.gaussian_value = norm(np.arange(-self.window_size, self.window_size+1, 1), scale = self.sd_gauss)
        
    def get_miss(self, gt, curr_id, next_id, a_space):
        seg_gt = gt[curr_id+1:next_id]
        total = len(seg_gt)
        n1 = sum(seg_gt)
        n0 = total - n1
        miss = (0.8 * n0 - n1) / max(a_space)
        return miss
    
    def get_acc(self, gt, next_id):
        try:
            seg_gt = gt[next_id - self.window_size : next_id + self.window_size + 1]
            gaussian_value = self.gaussian_value
        except:
            if next_id - self.window_size < 0:
                seg_gt = gt[:next_id + self.window_size + 1]
                gaussian_value = self.gaussian_value[-len(seg_gt):]
            else:
                seg_gt = gt[next_id - self.window_size : ]
                gaussian_value = self.gaussian_value[:len(seg_gt)]
        acc = np.sum(seg_gt * gaussian_value)
        return acc
    
    @staticmethod
    def forward(miss, acc, action):
        return miss + acc

class Agent(object):
    def __init__(self, learning_params, reward_estimator:RewardEstimator, action_space, Q_neural:NeuralNetwork, extract_model:Extractor = None, is_training = False):
        self.learning_params = learning_params
        self.reward_estimator = reward_estimator
        self.Q_neural = Q_neural
        self.extract_model = extract_model
        self.is_training = is_training
        self.is_feat_available = extract_model == None
        self.memory = Memo()
        
        self.a_space = action_space
        
        
    def load_data_video(self, feat, gt):
        self.feat = feat
        self.gt = gt
        self.num_frame = gt.shape[0]
    # or if run on raw video
    def load_video(self, video_capture: cv2.VideoCapture):
        self.vid_cap = video_capture
        self.num_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def apply_model(self, curr_id):
        if self.is_feat_available:
            curr_info, a_value =  self.Q_neural.forward(self.previous_info, self.feat[curr_id])
            return self.feat[curr_id], curr_info, a_value
        else:
            success, frame = self.vid_cap.read()
            if success:
                curr_feat = self.extract_model.forward(frame)
                curr_info = self.Q_neural.forward(self.previous_info, curr_feat)
                a_value = curr_info[-1]
                curr_info = curr_info[:-1]
                return curr_feat, curr_info, a_value

    def get_policy(self, a_value):
        if not self.is_training:
            a_index = np.argmax(a_value)
        else:
            exploration = np.random.choice(range(2),1,p=[1-self.learning_params.explore_rate,self.learning_params.explore_rate])
            if exploration==1:          # exploration
                a_index = np.random.choice(self.a_space,1)[0]
            else:                       # exploitation
                a_index = np.argmax(a_value)
        return a_index
    
    def get_action(self, a_index):
        action = self.a_space[a_index]
        return action
    
    def act(self, curr_id, action):
        if not self.is_feat_available:
            for _ in range(action-1):
                self.vid_cap.read()
        return curr_id + action
        
    def get_reward(self, curr_id, action, next_id):
        miss = self.reward_estimator.get_miss(self.gt, curr_id, next_id, self.a_space)
        acc = self.reward_estimator.get_acc(self.gt, next_id)
        return self.reward_estimator.forward(miss, acc, action)
    
    def update_action_value(self, r, next_feat, a_index, a_value, curr_info):
        target = a_value.copy()
        _, next_a_value = self.Q_neural.forward(curr_info, next_feat)
        max_next = max(next_a_value)
        target[a_index] = r + self.learning_params.decay_rate * max_next
        return target
    
    def train_batch(self):
        self.Q_neural.train(self.memory.pre_info_batch, self.memory.x_batch, self.memory.y_batch)
        self.memory.reset()
    
    def run_video(self, pre_handle_func = None, post_handle_func = None):
        curr_id = 0
        self.selection = np.zeros(self.num_frame)
        self.selection[curr_id]=1
        self.previous_info = self.Q_neural.init_info()
        while curr_id < self.num_frame:
            curr_feat, curr_info, a_value = self.apply_model(curr_id)
            a_index = self.get_policy(a_value)
            action = self.get_action(a_index)
            next_id = self.act(curr_id, action)
            if next_id >= self.num_frame:
                break
            self.selection[next_id]=1
            if self.is_training:
                r = self.get_reward()
                target_a_value = self.update_action_value()
                self.memory.add(self.previous_info, curr_feat, target_a_value)
                if self.memory.get_size() == self.learning_params.batch_size:
                    self.train_batch()
            self.previous_info = curr_info
            curr_id = next_id
            
    def save_summary(self, path, video_name):
        name = path + 'sum_' + video_name
        sio.savemat(name + '.mat',{'summary': self.selection})
        

        


        

    
    
    
    
    
    


    
    
    
    
        
