import os
import csv
from absl import flags
from absl import app

def get_csv_path(model_name):

    FLAGS = flags.FLAGS

    if FLAGS.use_pop_stats:
        eval_mode_str = 'eval'
    else:
        eval_mode_str = 'no_eval'
    
    if str(os.getcwd()).find('bitbucket') != -1:
        root_dir = './gpucluster/SVHN/'
    else:
        root_dir = './results/'

    csv_path_dir = root_dir + model_name + '/' + eval_mode_str + '/'  \
                    + FLAGS.attacks_in[0] + '/' 
    
    if FLAGS.relative_accuracy:
        acc_mode = 'relative'
    else:
        acc_mode = ''

    if FLAGS.test_noisy:
        if not os.path.isdir(csv_path_dir + 'noisy_test/'):
            os.mkdir(csv_path_dir + 'noisy_test/')
        csv_path_dir = csv_path_dir + 'noisy_test/'
                
    if len(os.listdir(csv_path_dir)) == 0:
        FLAGS.csv_path = csv_path_dir + model_name + '_' + FLAGS.dataset + '_' \
                            + 'results_' + eval_mode_str + + acc_mode + '.csv'
    elif len(os.listdir(csv_path_dir)) > 0:
        FLAGS.csv_path = csv_path_dir + model_name + '_' + FLAGS.dataset + '_' \
                            + 'results_' + eval_mode_str + '_' + acc_mode + '_adjusted' + '.csv'

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
        FLAGS.csv_path = csv_path_dir + model_name + '_' + FLAGS.dataset + '_' \
                            + 'results_' + eval_mode_str + '_' + acc_mode + '_' \
                            + noise_var_str + '_' + where_noise + '.csv' 
    
    return FLAGS.csv_path

def check_log(run_name, log_file):
    already_exists = False
    file = open(log_file)
    csvreader = csv.reader(file)
    for row in csvreader:
        if len(row) > 0 and str(row[0]) == run_name:
            print(run_name)
            already_exists = True
            break

    return already_exists


