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
flags.DEFINE_float('noise_variance', 0., 'value of injected noise variance')
flags.DEFINE_bool('noise_before_PGD', False, 'decide whether to use noise ')
flags.DEFINE_bool('noise_after_BN', None, 'decide if to apply noise before or after BN layer')
flags.DEFINE_bool('random_resizing', False, 'decide whether to apply random input resizing + cropping')
flags.DEFINE_bool('get_features', False, 'extract layer features mode')
flags.DEFINE_bool('adversarial_test', None, 'decide whether to test or not')
flags.DEFINE_bool('plot', False, 'decide whether to plot results or not')
flags.DEFINE_bool('use_pop_stats', None, 'decide whether to use .eval() mode')
flags.DEFINE_bool('no_eval_clean', False, 'for adversarial testing with clean batch stats')
flags.DEFINE_bool('get_logits', False, 'print decision logits')
flags.DEFINE_bool('save_to_log', False, 'save results to log')
flags.DEFINE_bool('save_to_wandb', False, 'save to weights and biases server')

flags.DEFINE_bool('noise_capacity_constraint', False, 'noise selection based on KL capacity constraint')
flags.DEFINE_bool('capacity_calculation', False, 'decide whether to calculate capacity or not')
flags.DEFINE_list('capacity', None, 'list of attacks to use')

flags.DEFINE_string('get_similarity', '', 'compute similarity based on the selection')
flags.DEFINE_bool('get_max_indexes', False, 'analysis involving CKA and capacity')

flags.DEFINE_bool('scaled_noise', False, 'decide whether to scale the input noise based on capacity or not')
flags.DEFINE_bool('scaled_noise_norm', False, 'decide whether to scale the input noise based on capacity or not')
flags.DEFINE_bool('scaled_noise_total', False, 'noise scaling based on total fixed capacity')
flags.DEFINE_bool('scaled_lambda', False, 'noise scaling based on total lambda value')
flags.DEFINE_bool('noise_first_layer', False, 'inject noise only at the very first layer')
flags.DEFINE_bool('noise_not_first_layer', False, 'inject noise evrywhere but at the very first layer')


flags.DEFINE_string('channel_transfer', '', 'implement feature transfer for testing')
flags.DEFINE_integer('n_channels_transfer', 0, 'number of channels to transfer')
flags.DEFINE_string('transfer_mode', '', 'decide which transfer mode to use')
flags.DEFINE_integer('layer_to_test', None, 'decide which layer to transfer activation to')

flags.DEFINE_bool('capacity_regularization', False, 'use capacity regularization')
flags.DEFINE_float('beta', 0.01, 'lagrangian multiplier for capacity regularization')
flags.DEFINE_string('regularization_mode', '', 'capacity mode to choose from: "gauss_entropy", "capacity", "lambda_entropy"')

flags.DEFINE_bool('get_saliency_map', False, 'get saliency maps')

flags.DEFINE_bool('frequency_analysis', False, 'carry out frequency analsysis of activations')
flags.DEFINE_integer('frequency_radius', 0, 'frequency radius to use for frequency decomposition')
flags.DEFINE_string('which_frequency', None, 'choose between low or high to test')
flags.DEFINE_bool('test_frequency', False, 'choose whether to test high/low frequency components')

flags.DEFINE_bool('IB_noise_calculation', False, 'carry out noise calculation via IB principle')

flags.DEFINE_bool('parametric_frequency_MSE', False, 'carry out noise std calculation for paramteric frequency MSE')
flags.DEFINE_bool('parametric_frequency_MSE_CE', False, 'carry out noise std calculation for paramteric frequency MSE+CE')

flags.DEFINE_bool('rank_init', False, 'use rank-preserving initialization')
flags.DEFINE_integer('pre_training_steps', 0, 'number of pre-training steps to use for rank-preserving initialization')

flags.DEFINE_bool('track_rank', False, 'track rank of last layer activations during training')

flags.DEFINE_bool('use_SkipInit', False, 'initialize ResNet with SkipInit')

flags.DEFINE_bool('train_small_lr', False, 'train models with BN with smaller LR than usual')

flags.DEFINE_bool('train_with_GaussianBlurr', False, 'train model with Gaussian Blurring')
flags.DEFINE_bool('train_with_low_frequency', False, 'train model with low-frequency data augmentation')

flags.DEFINE_bool('use_scaling', False, 'train model with scale layer')
flags.DEFINE_bool('test_low_pass_robustness', False, 'test low pass robustness')
flags.DEFINE_list('radii_to_test', [], 'radii to test')

flags.DEFINE_integer('low_pass_radius', None, 'radius for low pass filtering')
flags.DEFINE_bool('compare_frequency_domain', False, 'test for comparison of components in the frequency domain')
flags.DEFINE_bool('attenuate_HF', False, 'apply High-Frequency attenuation')

flags.DEFINE_bool('bounded_lambda', False, 'train with bounded bounded constraint')
flags.DEFINE_bool('nonlinear_lambda', False, 'train with bounded bounded constraint')
flags.DEFINE_bool('dropout_lambda', False, 'train with bounded bounded constraint')
flags.DEFINE_bool('free_lambda', False, 'train with bounded bounded constraint with non-trainable lambda but adaptive')
flags.DEFINE_bool('uniform_lambda', False, 'train with uniform lambda')

flags.DEFINE_bool('use_bn_scaling', False, 'test IB noise with BN scaling')
flags.DEFINE_bool('use_conv', False, 'use convultional activations for frequency analysis')

flags.DEFINE_bool('adversarial_transferrability', False, 'adversarial transferrability mode')
flags.DEFINE_string('model_name_to_attack', '', 'pretrained model name in adversarial_transferrability mode') 
flags.DEFINE_string('pretrained_name_to_attack', '', 'pretrained model run name in adversarial_transferrability mode') 

flags.DEFINE_string('prune_mode', '', 'prune mode')
flags.DEFINE_float('prune_percentage', 1., 'percentage of the channels to prune in a layer')

flags.DEFINE_bool('prune_mixed', False, 'prune model based on layer lambda value')
flags.DEFINE_bool('prune_random', False, 'prune model based on layer lambda value')

flags.DEFINE_bool('track_capacity', False, 'decide or not whether to track the progression of lambda during training')

flags.DEFINE_bool('large_epsilon', False, 'decide whether to test on large epsilon')

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
