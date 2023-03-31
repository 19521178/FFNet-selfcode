import time
from utils.math import sigmoid
from agent import Agent, RewardEstimator
from dataset import Dataset, TVSum
from evaluator import Evaluator, FScore
from layers import create_FC_layers
import numpy as np
from neural_network import FCNet, GRUNet
from utils.DictNamespace import DictNamespace as dict
from static_var import *

#############################################
# CONFIG
#############################################
opt_params = dict(
    optimizer = TypesOptimizer.RMSPROP,
    lr = 1e-4,
    loss = TypesLoss.MEAN_SQUARE
)

agent_learning_params = dict(
    batch_size = 128,
    exp_rate = 1,
    exp_low = 0.1,
    exp_decay = 0.00001,
    decay_rate = 0.8,
    action_space = np.arange(1, 26)
)

model_params = dict(
    dim_input = 4096,
    dim_hidden = 256,
    dim_output = 25
)

process_params = dict(
    max_eps = 1500,
    save_per_eps = 1,
    valid_per_eps = 10,
    savepath = '',
    filename = '',
)

# regular_params = dict(
#     type = TypesRegularizer.L2,
#     scale = 0.2
# )
regular_params = None

#############################################
# INIT DATASET
#############################################
dataset = TVSum(
    video_path='data/TVSum/video/',
    feat_path='data/TVSum/feat/',
    gt_path='data/TVSum/gt/'
)
test_name = ['J0nA4VgnoCo','vdmoEJ5YbrQ','0tmA_C6XwfM','Yi4Ij2NM7U4','XkqCExn6_Us','z_6gVvQb2d0','xmEERLqJ2kU','EE-bNr36nyA','eQu1rNs0an0','kLxoNp-UchI']
dataset.split_train_test(test_name=test_name)

#############################################
# INIT EVALUATOR
#############################################
evaluator = FScore()

#############################################
# INIT REWARD HANDLER
#############################################
reward_estimator = RewardEstimator(window_size=4, sd_gauss=1.)
def reward_func(miss, acc, action):
    return miss + acc - (sigmoid(action-1) - 0.5)
reward_estimator.forward = reward_func

#############################################
# INIT MODEL AND AGENT
#############################################
type_model = TypesModel.FC
dim_input = 4096
dims_FC = [400, 200, 100, 25]
FC_layers = create_FC_layers([dim_input] + dims_FC, is_regular=False, regular_params = regular_params)
model = FCNet(opt_params, model_params, FC_layers)
agent = Agent(learning_params = agent_learning_params, 
              reward_estimator=reward_estimator,
              action_space = agent_learning_params.action_space, 
              Q_neural = model, 
              is_training=True)

# type_model = TypesModel.GRU
# dim_input = 4096
# dim_hidden = 256
# dims_FC = [128, 64, 32, 25]
# FC_layers = create_FC_layers([dim_hidden] + dims_FC)
# model = GRUNet(opt_params, model_params, FC_layers)
# agent = Agent(agent_learning_params, model, reward_function, is_training=True)




#############################################
# DEFINE AND PREPARE FOR CONTINUE TRAINING OR JUST BEGIN
#############################################

# Load index current epoch from checkpoint
try:
    with open(process_params.savepath+'checkpoint', 'r') as f:
        lines = f.readlines()
    last_checkpoint = lines[0].split('_')[-1].split('"')[0]
    curr_epoch = int(last_checkpoint)
except:
    curr_epoch = 0   #0 if is_continue_train false

# Define explore-rate
try:
    with open(process_params.savepath+'log_exp_rate.txt', 'r') as f:
        lines = f.readlines()
    epoch_exprate = np.array([list(map(float, line[:-1].split(' '))) for line in lines])
    current_exprate = epoch_exprate[epoch_exprate[:, 0]==curr_epoch][-1, 1]
    agent.learning_params.explore_rate = current_exprate
except:
    agent.learning_params.explore_rate -= curr_epoch*0.0015
    agent.learning_params.explore_rate = max(agent.learning_params.explore_rate, agent.learning_params.explore_low)
    
# Load model for continue or save init model
if curr_epoch>0:
    agent.Q_neural.load_model(process_params.savepath, 'epoch_' + str(curr_epoch))
else:
    agent.Q_neural.save_model(process_params.savepath, 'epoch_' + str(curr_epoch))
    
    
    

#############################################
# TRAINING
#############################################
for epoch in range(curr_epoch, process_params):
    precisions = []
    recalls = []
    start_time = time.time()
    for video_index in range(len(dataset.train_name)):
        video_name, feat, gt = dataset.get_video_data(video_index, is_trainset=True)
        agent.load_data_video(feat, gt)
        agent.run_video()
        evaluator.load_label(video_name, gt, agent.selection)
        precision, recall, _ = evaluator.get_score_vid()
        precisions.append(precision)
        recalls.append(recall)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = evaluator.fscore_func(mean_precision, mean_recall)
    
    # Print and store Fscore of agent while learning
    print(evaluator.score_tostring([mean_precision, mean_recall, mean_f1]), time.time() - start_time)
    with open(process_params.savepath+'logs.txt', 'a') as f:
        f.write(evaluator.score_tostring([mean_precision, mean_recall, mean_f1]))
        f.write('\n')
        
    # Save model
    if ((epoch+1)%process_params.save_per_eps==0):
        agent.Q_neural.save_model(process_params.savepath, 'epoch_' + str(epoch+1))
        # agent.recover_target(savepath, filename+'_'+str(epoch+1))
        print('Save model at epoch '+ str(epoch+1)+'\tExplore Rate: '+str(agent.learning_params.explore_rate))
        with open(process_params.savepath+'log_exp_rate.txt', 'a') as f:
            f.write(str(epoch+1)+' '+str(agent.learning_params.explore_rate))
            f.write('\n')

    # Validate model on test dataset
    