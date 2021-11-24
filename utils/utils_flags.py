import numpy as np
from absl import flags
from torchvision.models.vgg import VGG, vgg16
from datetime import datetime
from utils.train_utils import train

# epsilon = np.arange(0.01, 0.2, 0.01)
epsilon = 0.1

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

# stadard flags
flags.DEFINE_string('device', 'cuda:0', 'device name - alernatives: "cpu"')
flags.DEFINE_string('model_name', None, 'model name among available models')
flags.DEFINE_integer('batch_size', 128, 'batch size for training')
flags.DEFINE_string('dataset', 'CIFAR10', 'dataset to use')
flags.DEFINE_string('dataset_path', '/data2/users/lr4617/data/', 'path to dataset')
flags.DEFINE_string('root_path', '/data2/users/lr4617', 'path to root directory')

flags.DEFINE_bool('train', True, 'decide whether to train or not')
flags.DEFINE_bool('load_pretrained', False, 'decide whether to load pre trained model')
flags.DEFINE_bool('test', True, 'decide whether to test or not')
flags.DEFINE_bool('adversarial_test', True, 'decide whether to test or not')
flags.DEFINE_bool('save_to_log', True, 'save results to log')

flags.DEFINE_string('mode', None, 'training mode to use')
flags.DEFINE_string('attack', None, 'type of adversarial attack')
flags.DEFINE_list('epsilon', [epsilon], 'espilon range over which to test adversarial robustness')
flags.DEFINE_bool('test_run', False, 'variable for running test run of some new feature')
flags.DEFINE_list('where_bn', None, 'variable for deciding where to put BN layer')
flags.DEFINE_list('bn_locations', None, 'variable for deciding where to put BN layer')
flags.DEFINE_string('pretrained_name', None, 'run name for existing trained model')


flags.mark_flag_as_required('model_name')
flags.mark_flag_as_required('train')
flags.mark_flag_as_required('load_pretrained')
flags.mark_flag_as_required('pretrained_name')
flags.mark_flag_as_required('bn_locations')

FLAGS = flags.FLAGS
