import numpy as np
from absl import flags
from datetime import datetime
from utils.train_utils import train

# epsilon = np.arange(0.01, 0.2, 0.01)
epsilon = 0.0313

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

# stadard flags
flags.DEFINE_string('device', None, 'device name - alernatives: "cpu"')
flags.DEFINE_string('model_name', None, 'model name among available models')
flags.DEFINE_integer('version', None, 'model version to use (only for ResNet)')
flags.DEFINE_integer('batch_size', 128, 'batch size for training')
flags.DEFINE_string('dataset', None, 'dataset to use')
flags.DEFINE_string('dataset_path', '/data2/users/lr4617/data/', 'path to dataset')
flags.DEFINE_string('root_path', '/data2/users/lr4617', 'path to root directory')

flags.DEFINE_bool('train', None, 'decide whether to train or not')
flags.DEFINE_bool('load_pretrained', None, 'decide whether to load pre trained model')
flags.DEFINE_bool('test', None, 'decide whether to test or not')
flags.DEFINE_bool('adversarial_test', None, 'decide whether to test or not')
flags.DEFINE_bool('save_to_log', None, 'save results to log')

flags.DEFINE_string('mode', None, 'training mode to use')
flags.DEFINE_string('attack', None, 'type of adversarial attack')
flags.DEFINE_list('epsilon', None, 'espilon range over which to test adversarial robustness')
flags.DEFINE_list('attacks_in', ['FGSM', 'PGD'], 'attacks defined by the user')
flags.DEFINE_bool('test_run', False, 'variable for running test run of some new feature')
flags.DEFINE_list('where_bn', None, 'variable for deciding where to put BN layer')
flags.DEFINE_integer('bn_locations', None, 'variable for deciding where to put BN layer')
flags.DEFINE_string('pretrained_name', None, 'run name for existing trained model')

flags.mark_flag_as_required('model_name')
flags.mark_flag_as_required('train')
flags.mark_flag_as_required('load_pretrained')
flags.mark_flag_as_required('pretrained_name')
flags.mark_flag_as_required('bn_locations')

FLAGS = flags.FLAGS
