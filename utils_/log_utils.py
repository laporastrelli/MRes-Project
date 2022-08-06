import os
import csv
from absl import flags
from absl import app

FLAGS = flags.FLAGS

def get_csv_path(model_name):

    ##################### DEFINE DIRECTORIES #####################
    # root directory based on server
    if str(os.getcwd()).find('bitbucket') != -1:
        if FLAGS.dataset=='SVHN':
            root_dir = './gpucluster/SVHN/'
        elif FLAGS.dataset=='CIFAR10':
            root_dir = './gpucluster/CIFAR10/'
        elif FLAGS.dataset=='CIFAR100':
            root_dir = './gpucluster/CIFAR100/'
    else:
        root_dir = './results/'

    # eval or no-eval mode
    if FLAGS.use_pop_stats:
        eval_mode_str = 'eval'
    else:
        eval_mode_str = 'no_eval'
    
    # directory before analysis
    csv_path_dir = root_dir + model_name + '/' + eval_mode_str + '/'  \
                    + FLAGS.attacks_in[0] + '/' 
    if not os.path.isdir(csv_path_dir):
        os.mkdir(csv_path_dir)
    
    if FLAGS.test_frequency:
        if not os.path.isdir(csv_path_dir + 'frequency_test/'):
            os.mkdir(csv_path_dir + 'frequency_test/')
        csv_path_dir = csv_path_dir + 'frequency_test/'

    # nosiy test
    if FLAGS.test_noisy:
        if not os.path.isdir(csv_path_dir + 'noisy_test/'):
            os.mkdir(csv_path_dir + 'noisy_test/')
        csv_path_dir = csv_path_dir + 'noisy_test/'
    
    # capacity regularization  
    if FLAGS.capacity_regularization:
        if not os.path.isdir(csv_path_dir + 'capacity_regularization/'):
            os.mkdir(csv_path_dir + 'capacity_regularization/')
        csv_path_dir = csv_path_dir + 'capacity_regularization/'
    
    # rank init
    if FLAGS.rank_init:
        if not os.path.isdir(csv_path_dir + 'rank_init/'):
            os.mkdir(csv_path_dir + 'rank_init/')
        csv_path_dir = csv_path_dir + 'rank_init/' 
    
    if FLAGS.use_SkipInit:
        if not os.path.isdir(csv_path_dir + 'use_SkipInit/'):
            os.mkdir(csv_path_dir + 'use_SkipInit/')
        csv_path_dir = csv_path_dir + 'use_SkipInit/'
    
    if FLAGS.normalization == 'ln':
        if not os.path.isdir(csv_path_dir + 'train_ln/'):
            os.mkdir(csv_path_dir + 'train_ln/')
        csv_path_dir = csv_path_dir + 'train_ln/'
    
    if FLAGS.train_small_lr:
        if not os.path.isdir(csv_path_dir + 'train_small_lr/'):
            os.mkdir(csv_path_dir + 'train_small_lr/')
        csv_path_dir = csv_path_dir + 'train_small_lr/'
    
    if FLAGS.train_with_GaussianBlurr:
        if not os.path.isdir(csv_path_dir + 'train_with_GaussianBlurr/'):
            os.mkdir(csv_path_dir + 'train_with_GaussianBlurr/')
        csv_path_dir = csv_path_dir + 'train_with_GaussianBlurr/'

    if FLAGS.train_with_low_frequency:
        if not os.path.isdir(csv_path_dir + 'train_with_low_frequency/'):
            os.mkdir(csv_path_dir + 'train_with_low_frequency/')
        csv_path_dir = csv_path_dir + 'train_with_low_frequency/'
    
    if FLAGS.use_scaling:
        if not os.path.isdir(csv_path_dir + 'use_scaling/'):
            os.mkdir(csv_path_dir + 'use_scaling/')
        csv_path_dir = csv_path_dir + 'use_scaling/'
    
    # test low pass robustness
    if FLAGS.test_low_pass_robustness:
        if not os.path.isdir(csv_path_dir + 'test_low_pass_robustness/'):
            os.mkdir(csv_path_dir + 'test_low_pass_robustness/')
        csv_path_dir = csv_path_dir + 'test_low_pass_robustness/' 

    # channel transfer
    if len(FLAGS.channel_transfer) > 0 :
        if not os.path.isdir(csv_path_dir + 'channel_transfer/'):
            os.mkdir(csv_path_dir + 'channel_transfer/')
        csv_path_dir = csv_path_dir + 'channel_transfer/'
        FLAGS.csv_path = csv_path_dir + model_name + '_' + FLAGS.dataset + '_' \
                            + 'results_' + eval_mode_str + '_' + acc_mode + FLAGS.channel_transfer +  '.csv'

    if FLAGS.attenuate_HF:
        if not os.path.isdir(csv_path_dir + 'attenuate_HF/'):
            os.mkdir(csv_path_dir + 'attenuate_HF/')
        csv_path_dir = csv_path_dir + 'attenuate_HF/' 

    if FLAGS.bounded_lambda:
        if not os.path.isdir(csv_path_dir + 'bounded_lambda/'):
            os.mkdir(csv_path_dir + 'bounded_lambda/')
        csv_path_dir = csv_path_dir + 'bounded_lambda/' 
        if FLAGS.free_lambda:
            if not os.path.isdir(csv_path_dir + 'free_lambda/'):
                os.mkdir(csv_path_dir + 'free_lambda/')
            csv_path_dir = csv_path_dir + 'free_lambda/'
    
    if FLAGS.nonlinear_lambda:
        if not os.path.isdir(csv_path_dir + 'nonlinear_lambda/'):
            os.mkdir(csv_path_dir + 'nonlinear_lambda/')
        csv_path_dir = csv_path_dir + 'nonlinear_lambda/'
    
    if FLAGS.dropout_lambda:
        if not os.path.isdir(csv_path_dir + 'dropout_lambda/'):
            os.mkdir(csv_path_dir + 'dropout_lambda/')
        csv_path_dir = csv_path_dir + 'dropout_lambda/'

    ##################### DEFINE FILE NAME #####################
    if FLAGS.relative_accuracy:
        acc_mode = 'relative'
    else:
        acc_mode = ''

    if model_name.find('ResNet')!= -1 and FLAGS.version != 1:
        model_name += '_v' + str(FLAGS.version)

    if len(os.listdir(csv_path_dir)) == 0:
        FLAGS.csv_path = csv_path_dir + model_name + '_' + FLAGS.dataset + '_' \
                            + 'results_' + eval_mode_str + '_' + acc_mode + '.csv'
    elif len(os.listdir(csv_path_dir)) > 0:
        FLAGS.csv_path = csv_path_dir + model_name + '_' + FLAGS.dataset + '_' \
                            + 'results_' + eval_mode_str + '_' + acc_mode + '_adjusted' + '.csv'
    if FLAGS.test_frequency:
        FLAGS.csv_path = csv_path_dir + model_name + '_' + FLAGS.dataset + '_' \
                            + 'results_' + eval_mode_str + '_' + acc_mode + '_' \
                            + str(FLAGS.which_frequency) + '.csv'

    if FLAGS.test_noisy:
        noise_var_str = str(FLAGS.noise_variance).replace('.', '')
        if FLAGS.noise_after_BN:
            where_noise = 'after'
        else:
            where_noise = 'before'
        if not FLAGS.noise_before_PGD:
            where_noise += '_after_attack'
        if FLAGS.scaled_noise:
            where_noise += '_scaled'
        elif FLAGS.scaled_noise_norm:
            where_noise += '_scaled_norm'
        elif FLAGS.scaled_noise_total:
            where_noise += '_scaled_total'
        FLAGS.csv_path = csv_path_dir + model_name + '_' + FLAGS.dataset + '_' \
                            + 'results_' + eval_mode_str + '_' + acc_mode + '_' \
                            + noise_var_str + '_' + where_noise + '.csv' 
    
    if FLAGS.capacity_regularization:
        FLAGS.csv_path = csv_path_dir + model_name + '_' + FLAGS.dataset + '_' \
                            + 'results_' + eval_mode_str + '_' + acc_mode + '_' \
                            + str(FLAGS.regularization_mode) + '.csv'

    return FLAGS.csv_path

def check_log(run_name, log_file):
    already_exists = False
    if os.path.isfile(log_file):
        file = open(log_file)
        csvreader = csv.reader(file)
        for row in csvreader:
            if len(row) > 0 and str(row[0]) == run_name:
                print(run_name)
                already_exists = True
                break

    return already_exists

def get_csv_keys(csv_path, key=''):
    keys = []
    file = open(csv_path)
    csvreader = csv.reader(file)
    for row in csvreader:
        if len(row) > 0 and str(row[0]).find(key)!=-1:
            keys.append(str(row[0]))

    return keys




