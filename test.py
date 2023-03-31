from agent import Agent, RewardEstimator
from dataset import TVSum, Tour20
from evaluator import Coverage, FScore
from layers import create_FC_layers
import numpy as np
from neural_network import FCNet, GRUNet
from utils.DictNamespace import DictNamespace as dict
from static_var import *
from post_processor import ExtendAll, NoneExtend
from utils.math import sigmoid


opt_params = dict(
    optimizer = TypesOptimizer.RMSPROP,
    lr = 1e-4,
    loss = TypesLoss.MEAN_SQUARE
)

agent_learning_params = dict(
    action_space = np.arange(1, 26)
)

model_params = dict(
    dim_input = 4096,
    dim_hidden = 256,
    dim_output = 25
)

process_params = dict(
    savepath = '',
    loadpath = 'model/basemodel_newreward/',
    filename = 'epoch',
)

# regular_params = dict(
#     type = TypesRegularizer.L2,
#     scale = 0.2
# )
regular_params = None

reward_estimator = RewardEstimator(window_size=4, sd_gauss=1.)
def reward_func(miss, acc, action):
    return miss + acc - (sigmoid(action-1) - 0.5)
reward_estimator.forward = reward_func

type_model = TypesModel.FC
dim_input = 4096
dims_FC = [400, 200, 100, 25]
FC_layers = create_FC_layers([dim_input] + dims_FC, is_regular=False, regular_params = regular_params)
model = FCNet(opt_params, model_params, FC_layers)
agent = Agent(learning_params = agent_learning_params, 
              reward_estimator=reward_estimator,
              action_space = agent_learning_params.action_space, 
              Q_neural = model, 
              is_training=False)
print('Crafted Model')

# type_model = TypesModel.GRU
# dim_input = 4096
# dim_hidden = 256
# dims_FC = [64, 25]
# FC_layers = create_FC_layers([dim_hidden] + dims_FC)
# model = GRUNet(opt_params, model_params, FC_layers)
# agent = Agent(learning_params = agent_learning_params, 
#               reward_estimator=reward_estimator,
#               action_space = agent_learning_params.action_space, 
#               Q_neural = model, 
#               is_training=True)

dataset = TVSum(
    video_path='data/TVSum/video/',
    feat_path='data/TVSum/feat/',
    gt_path='data/TVSum/gt/'
)
test_name = ['J0nA4VgnoCo','vdmoEJ5YbrQ','0tmA_C6XwfM','Yi4Ij2NM7U4','XkqCExn6_Us','z_6gVvQb2d0','xmEERLqJ2kU','EE-bNr36nyA','eQu1rNs0an0','kLxoNp-UchI']
dataset.split_train_test(test_name=test_name)
print('Created dataset')

# dataset = Tour20(
#     video_path = '',
#     feat_path = '',
#     gt_path = ''
# )
# dataset.split_train_test()

fscore_evaluator = FScore()
fscore_evaluator.load_dataset(dataset)
hn_space = [i for i in range(1, 21)]
coverage_evaluator = Coverage(hn_space=np.arange(1, 21))
coverage_evaluator.load_dataset(dataset)
dataset.load_gt_segments()
# dataset.load_gt_segments(segment_path='')
print('Created evaluator')



none_post = NoneExtend()
w2_post = ExtendAll(window_size=2, step_select=1)
print('Created post processor')

for epoch in [860]:
    agent.Q_neural.load_model(process_params.loadpath, process_params.filename + '_' + str(epoch))
    print('Loaded model epoch', epoch)
    
    fscore_evaluator.reset_score_epoch()
    coverage_evaluator.reset_score_epoch()
    for video_index in range(len(dataset.test_name)):
        video_name, feat, gt = dataset.get_video_data(video_index, is_trainset=False)
        agent.load_data_video(feat, gt)
        agent.run_video()
        fscore_evaluator.load_label(video_name, gt, agent.selection)
        coverage_evaluator.load_label(video_name, gt, agent.selection)
        
        fscore_evaluator.get_score_vid(none_post)
        coverage_evaluator.get_score_vid(w2_post)
    
    val_fscore = fscore_evaluator.get_score_epoch()
    val_score_coverage_per_hn = coverage_evaluator.get_score_epoch()
    
    print(fscore_evaluator.score_tostring(val_fscore))
    print(coverage_evaluator.score_tostring(val_score_coverage_per_hn))
        