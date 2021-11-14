import numpy as np
from absl import flags
from torchvision.models.vgg import VGG, vgg16

from utils.train_utils import train

epsilon = np.arange(0.01, 0.2, 0.01)
# epsilon = [0.1]

flags.DEFINE_string('device', 'cuda:0', 'device name - alernatives: "cpu"')
flags.DEFINE_list('epsilon', epsilon.tolist(), 'espilon range over which to test adversarial robustness')
flags.DEFINE_boolean('batch_norm', None, 'use or not Batch Normalization')
flags.DEFINE_string('model_name', 'VGG19', 'model name among available models')
flags.DEFINE_integer('n_epochs', 75, 'number of epochs for training')
flags.DEFINE_integer('batch_size', 128, 'batch size for training')
flags.DEFINE_float('learning_rate', None, 'learning rate for optimizer')
flags.DEFINE_string('dataset', 'CIFAR10', 'dataset to use')
flags.DEFINE_string('dataset_path', '/data2/users/lr4617/data/', 'path to dataset')
flags.DEFINE_string('root_path', '/data2/users/lr4617', 'path to root directory')
flags.DEFINE_bool('train', True, 'decide whether to train or not')
flags.DEFINE_bool('test', True, 'decide whether to test or not')
flags.DEFINE_bool('adversarial_test', False, 'decide whether to test or not')
flags.DEFINE_bool('cross_mode_adv_test', False, 'decide whether to test or not')
flags.DEFINE_string('attack', 'FGSM', 'type of adversarial attack')

FLAGS = flags.FLAGS
