from enum import Enum

class TypesModel(Enum):
    FC = 0
    GRU = 1
    LSTM = 2


class TypesActivation(Enum):
    RELU = 'relu'
    LINEAR = 'linear'
    TANH = 'tanh'
    SIGMOID = 'sigmoid'
    SOFTMAX = 'softmax'
    LEAKYRELU = 'leaky_relu'


class TypesOptimizer(Enum):
    RMSPROP = 0
    SGD = 1
    Adam = 2


class TypesLoss(Enum):
    MEAN_SQUARE = 0
    CROSS_ENTROPY = 1


class TypesRegularizer(Enum):
    L1 = 0
    L2 = 1
    L1_L2 = 2



