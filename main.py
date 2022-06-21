############ IMPORTS ############
from tabnanny import check
from webbrowser import get
import os
import numpy as np
import csv
import torch
import wandb

from torch.utils.tensorboard import SummaryWriter
from utils_ import utils_flags
from absl import app
from absl import flags
from functions.train import train
from functions.test import test
from get_FAB_acc import get_FAB_acc
from utils_.miscellaneous import get_epsilon_budget, get_bn_int_from_name, get_bn_config_train, set_load_pretrained
from utils_.set_test_run import set_test_run
from utils_.log_utils import get_csv_path, check_log

def main(argv):
    
    del argv
    
    ######################################################### SETUP #########################################################

    # parse inputs 
    FLAGS = flags.FLAGS
    
    print(FLAGS.result_log)

    already_exists = False

    print(FLAGS.device)

    # set root paths depending on the server in use
    if str(os.getcwd()).find('bitbucket') != -1:
        FLAGS.root_path = '/vol/bitbucket/lr4617'
        FLAGS.dataset_path = '/vol/bitbucket/lr4617/data/'
    else:
        FLAGS.root_path = '/data2/users/lr4617'
        FLAGS.dataset_path = '/data2/users/lr4617/data/' 

    # get device (for gpucluster)
    if FLAGS.device is None:
        FLAGS.device = 'cuda:' + str(torch.cuda.current_device())
        FLAGS.device = 'cuda:0'

    # retrive dataset-corresponding epsilon budget
    FLAGS.epsilon_in = get_epsilon_budget(dataset=FLAGS.dataset)
    if FLAGS.test_noisy: FLAGS.epsilon_in = FLAGS.epsilon_in[0:3]
    if FLAGS.adversarial_test and FLAGS.model_name.find('ResNet')!= -1: FLAGS.epsilon_in = FLAGS.epsilon_in[0:3]
    
    # model name logistics
    if FLAGS.model_name.find('ResNet50_v') != -1: FLAGS.model_name = 'ResNet50'        

    # get BN locations from pretrained model name (for testing only)
    if FLAGS.load_pretrained:
        if FLAGS.result_log.find(',') != -1: result_log = FLAGS.result_log.split(',')
        elif FLAGS.result_log.find(';') != -1: result_log = FLAGS.result_log.split(';')
        FLAGS.bn_locations = get_bn_int_from_name(run_name=FLAGS.pretrained_name)
    
    # get model name, based on it determine one-hot encoded BN locations 
    model_name = FLAGS.model_name
    bn_locations = get_bn_config_train(model_name=FLAGS.model_name, bn_int=FLAGS.bn_locations)
    FLAGS.load_pretrained = set_load_pretrained(FLAGS.train, FLAGS.test_run)

    # define test run params
    if FLAGS.test_run: index, bn_string, test_acc, adv_accs, result_log = set_test_run()
    
    # dict selection based on mode
    if FLAGS.capacity_regularization:
        FLAGS.epsilon_in = FLAGS.epsilon_in[0:2]
        if FLAGS.save_to_log:
            columns_csv = ['Run', 'Model', 'Dataset', 'Batch-Normalization', 
                           'Training Mode', 'beta-lagrange', 'Test Accuracy', 'Epsilon Budget']
    else:
        if FLAGS.save_to_log:
            columns_csv = ['Run', 'Model', 'Dataset', 'Batch-Normalization', 
                           'Training Mode', 'Test Accuracy', 'Epsilon Budget'] 

    # carry out channel transfer only for full-BN configs
    if FLAGS.channel_transfer:
        if get_bn_int_from_name(FLAGS.pretrained_name)!= 100: 
            already_exists = True

    # save to results log if file not already saved
    if FLAGS.save_to_log:
        csv_path = get_csv_path(FLAGS.model_name)
        if FLAGS.load_pretrained: 
            already_exists = check_log(run_name=FLAGS.pretrained_name, log_file=csv_path)
            print('ALREAD EXISTS IN RESULTS LOG: ', already_exists)

    # display model info
    if FLAGS.verbose:
        print('-----------------------------------------------------------------------------')
        print('| Model Name:         ', model_name)
        print('| Dataset:            ', FLAGS.dataset)
        print('| BN Configuration:   ', bn_locations)
        print('| Train:              ', FLAGS.train)
        print('| Test:               ', FLAGS.test)
        print('| Test (with noise):  ', FLAGS.test_noisy)
        if FLAGS.test_noisy:
           print('| Noise after BN:     ', FLAGS.noise_after_BN)             
        print('| Adversarial Test:   ', FLAGS.adversarial_test)
        if FLAGS.adversarial_test:
            print('| Attack:             ', FLAGS.attacks_in[0])
            print('| Epsilon Budget:     ', FLAGS.epsilon_in)
        print('-----------------------------------------------------------------------------')
        print('RUN: ', not already_exists)

    ######################################################### OPERATIONS #########################################################

    where_bn = bn_locations
    
    if FLAGS.train:
        if not FLAGS.test_run:
            if FLAGS.save_to_wandb:
                index, run = train(model_name, where_bn)
            else:
                index = train(model_name, where_bn)
            result_log=[]

    if not already_exists:
        
        if FLAGS.load_pretrained:
            index = FLAGS.pretrained_name

        if FLAGS.test:
            test_acc = test(index, standard=True)

        if FLAGS.get_saliency_map:
            _ = test(index, get_saliency_map=True)
        
        if FLAGS.channel_transfer:
            _ = test(index, channel_transfer=True) 

        if FLAGS.get_features:
            if FLAGS.adversarial_test:
                for attack in FLAGS.attacks_in:
                    FLAGS.attack = attack
            if FLAGS.adversarial_test:
                for attack in FLAGS.attacks_in:
                    FLAGS.attack = attack
                    for eps in FLAGS.epsilon_in:
                        FLAGS.epsilon = float(eps)
                        _ = test(index, get_features=True, adversarial=True)
            else:
                _ = test(index, get_features=True)
            
            # setting adversarial_test back to False
            FLAGS.adversarial_test = False

        if FLAGS.adversarial_test or FLAGS.capacity_calculation:
            adv_accs = dict()
            for attack in FLAGS.attacks_in:
                FLAGS.attack = attack
                if attack == 'PGD':
                    for eps in FLAGS.epsilon_in:
                        FLAGS.epsilon = float(eps)
                        dict_name = attack + '-' + str(FLAGS.epsilon)
                        if FLAGS.capacity_calculation:
                            _ =  test(index, capacity_calculation=True)
                        else:
                            adv_accs[dict_name] = test(index, adversarial=True)
                        
                elif attack in ['FAB', 'APGD_CE', 'APGD_DLR', 'Square', '-PGD']:
                    if FLAGS.adversarial_test:
                        for eps in FLAGS.epsilon_in:
                            FLAGS.epsilon = float(eps)
                            dict_name = attack + '-' + str(FLAGS.epsilon)
                            adv_accs[dict_name] = get_FAB_acc(index, attack)

        if FLAGS.save_to_log:
            model_name_ = FLAGS.model_name
            if FLAGS.model_name.find('ResNet')!=-1 and FLAGS.version!=1:
                model_name_ = FLAGS.model_name + '_v' + str(FLAGS.version)
            
            if sum(where_bn)==0:
                bn_string = 'No'
            elif sum(where_bn)>1:
                bn_string = 'Yes - ' + 'all'
            else:
                bn_string = 'Yes - ' + str(where_bn.index(1) + 1) + ' of ' + str(len(where_bn))

            if FLAGS.train and len(result_log)<=1:
                if FLAGS.capacity_regularization:
                    csv_dict = {
                        columns_csv[0] : index,
                        columns_csv[1] : model_name_,
                        columns_csv[2] : FLAGS.dataset,
                        columns_csv[3] : bn_string, 
                        columns_csv[4] : FLAGS.mode,
                        columns_csv[5] : FLAGS.beta,
                        columns_csv[6] : test_acc,
                        columns_csv[7] : FLAGS.epsilon_in}
                else:
                    csv_dict = {
                        columns_csv[0] : index,
                        columns_csv[1] : model_name_,
                        columns_csv[2] : FLAGS.dataset,
                        columns_csv[3] : bn_string, 
                        columns_csv[4] : FLAGS.mode,
                        columns_csv[5] : test_acc,
                        columns_csv[6] : FLAGS.epsilon_in}
                csv_dict.update(adv_accs)

            elif len(result_log)>1 and not FLAGS.capacity_regularization:    
                csv_dict = dict()
                for i, log in enumerate(result_log):
                    if i <=5:
                        csv_dict[columns_csv[i]] = log
                if FLAGS.test:
                    csv_dict[columns_csv[5]] = test_acc
                csv_dict.update(adv_accs)    
            
            elif len(result_log)>1 and FLAGS.capacity_regularization:
                csv_dict = dict()
                for i, log in enumerate(result_log):
                    if i <=6:
                        csv_dict[columns_csv[i]] = log
                if FLAGS.test:
                    csv_dict[columns_csv[6]] = test_acc
                csv_dict.update(adv_accs)  

            if FLAGS.save_to_wandb: 
                try: 
                    results_table = wandb.Table(columns=list(csv_dict.keys()), data=list(csv_dict.values()))
                    run.log({"results_table": results_table})
                except Exception as e:
                    print("Error (wandb-custom): trouble saving table to wandb")
                
            try:
                FLAGS.csv_path = get_csv_path(model_name)             
                with open(FLAGS.csv_path, 'a') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_dict.keys())
                    writer.writerow(csv_dict)
            except IOError:
                print("I/O error")

        if FLAGS.plot:
            if sum(where_bn)==0:
                bn_string = '0'
            elif sum(where_bn)>1:
                bn_string = '100'
            else:
                bn_string = str(where_bn.index(1) + 1)
            
            root = '/data2/users/lr4617/An_Information_Theoretic_View_of_BN/adversarial_ml/MRes-Project'

            name = FLAGS.model_name + '_' + bn_string
            folder_name = FLAGS.which
            print(result_log)
            print(result_log[7:])
            to_save = np.asarray(result_log[7:]).astype(np.float64)
            path_out = root + '/plot/' + folder_name + '/' + name +  '_' + FLAGS.which + '.npy'

            print(name)

            if os.path.isfile(path_out):
                temp = np.load(path_out)
                out = np.vstack((temp, to_save))
            else:
                out = to_save

            np.save(path_out, out)

if __name__ == '__main__':
    app.run(main)
