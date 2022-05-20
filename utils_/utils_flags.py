from curses import flash
import numpy as np
from absl import flags
from datetime import datetime
from utils_.train_utils import train

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

flags.DEFINE_string('device', None, 'device name')
flags.DEFINE_string('model_name', None, 'model name among available models')
flags.DEFINE_integer('version', None, 'model version to use (only for ResNet)')
flags.DEFINE_integer('batch_size', 128, 'batch size for training')
flags.DEFINE_string('dataset', None, 'dataset to use')
flags.DEFINE_string('root_path', '/vol/bitbucket/lr4617', 'path to root directory')
flags.DEFINE_string('dataset_path', '/vol/bitbucket/lr4617/data/', 'path to dataset')
flags.DEFINE_string('csv_path', None, 'path to csv results file')
flags.DEFINE_bool('verbose', True, 'Prompt informative text about running session')

flags.DEFINE_bool('train', None, 'decide whether to train or not')
flags.DEFINE_bool('train_noisy', None, 'decide whether to train with noise or not')
flags.DEFINE_float('train_noise_variance', None, 'noise variance used during training (in case noisy training is chosen)')
flags.DEFINE_bool('load_pretrained', None, 'decide whether to load pre trained model')
flags.DEFINE_bool('test', None, 'decide whether to test or not')
flags.DEFINE_bool('relative_accuracy', False, 'evaluate with respect to correct batch correct samples')
flags.DEFINE_bool('test_noisy', False, 'inject noise at test time')
flags.DEFINE_float('noise_variance', 0, 'value of injected noise variance')
flags.DEFINE_bool('noise_before_PGD', False, 'decide whether to use noise ')
flags.DEFINE_bool('noise_after_BN', None, 'decide if to apply noise before or after BN layer')
flags.DEFINE_bool('random_resizing', False, 'decide whether to apply random input resizing + cropping')
flags.DEFINE_bool('get_features', False, 'extract layer features mode')
flags.DEFINE_bool('adversarial_test', None, 'decide whether to test or not')
flags.DEFINE_bool('plot', False, 'decide whether to plot results or not')
flags.DEFINE_bool('use_pop_stats', False, 'decide whether to use .eval() mode')
flags.DEFINE_bool('no_eval_clean', False, 'for adversarial testing with clean batch stats')
flags.DEFINE_bool('get_logits', False, 'print decision logits')
flags.DEFINE_bool('save_to_log', False, 'save results to log')
flags.DEFINE_bool('noise_capacity_constraint', False, 'noise selection based on KL capacity constraint')
flags.DEFINE_bool('capacity_calculation', False, 'decide whether to calculate capacity or not')
flags.DEFINE_list('capacity', None, 'list of attacks to use')
flags.DEFINE_bool('scaled_noise', False, 'decide whether to scale the input noise based on capacity or not')

flags.DEFINE_string('normalization', 'bn', 'normalization to use')
flags.DEFINE_string('mode', None, 'training mode to use')
flags.DEFINE_string('attack', None, 'type of adversarial attack')
flags.DEFINE_list('attacks_in', ['PGD'], 'list of attacks to use')
flags.DEFINE_float('epsilon', None, 'espilon range over which to test adversarial robustness')
flags.DEFINE_list('epsilon_in', None, 'espilon range over which to test adversarial robustness')
flags.DEFINE_integer('PGD_iterations', None, 'number of iteration to run PGD for')
flags.DEFINE_list('where_bn', None, 'variable for deciding where to put BN layer')
flags.DEFINE_integer('bn_locations', None, 'variable for deciding where to put BN layer')
flags.DEFINE_string('pretrained_name', None, 'run name for existing trained model')
flags.DEFINE_string('result_log', None, 'dict as string for result log parsing')
flags.DEFINE_string('which', None, 'which mode to save')
flags.DEFINE_list('input_size', None, 'temp')


# run this to test script logistics
flags.DEFINE_bool('test_run', False, 'variable for running test run of some new feature')

FLAGS = flags.FLAGS
