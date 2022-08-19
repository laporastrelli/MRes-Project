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
    
    ######################################## SETUP #########################################
    # parse inputs 
    FLAGS = flags.FLAGS
    
    already_exists = False

    print('capacity_regularization: ', FLAGS.capacity_regularization)
    print('device: ', FLAGS.device)
    print('BATCH SIZE: ', FLAGS.batch_size)

    print('Pruning Mode: ', FLAGS.prune_mode)
    print('Layer to Prune: ', FLAGS.layer_to_test)

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
    ################################################################################################

    ####################################### EPSILON BUDGET #######################################
    # retrive dataset-corresponding epsilon budget
    FLAGS.epsilon_in = get_epsilon_budget(dataset=FLAGS.dataset)
    # retrieve noisy-mode-corresponding epsilon budget
    if FLAGS.test_noisy: 
        FLAGS.epsilon_in = FLAGS.epsilon_in[0:4]
        if get_bn_int_from_name(FLAGS.pretrained_name) not in [100]:
            already_exists = True
    # retrieve eval-mode-corresponding epsilon budget
    if FLAGS.adversarial_test: 
        if FLAGS.use_pop_stats:
            FLAGS.epsilon_in = FLAGS.epsilon_in[0:4]
    # retrieve frequency-mode-corresponding epsilon budget
    if FLAGS.test_low_pass_robustness:
        FLAGS.epsilon_in = FLAGS.epsilon_in[0:5]
    # retrieve adversarial_transferrability-mode-corresponding epsilon budget
    if FLAGS.adversarial_transferrability:
        FLAGS.epsilon_in = [FLAGS.epsilon_in[0]]
    ################################################################################################

    ####################################### Logistics #############################################
    # model name logistics
    if FLAGS.model_name.find('ResNet50_v') != -1: FLAGS.model_name = 'ResNet50'        
    # get BN locations from pretrained model name (for testing only)
    if FLAGS.load_pretrained:
        if FLAGS.result_log.find(',') != -1: result_log = FLAGS.result_log.split(',')
        elif FLAGS.result_log.find(';') != -1: result_log = FLAGS.result_log.split(';')
        FLAGS.bn_locations = get_bn_int_from_name(run_name=FLAGS.pretrained_name)
    # get model name, based on it determine one-hot encoded BN locations 
    model_name = FLAGS.model_name
    print(FLAGS.model_name, FLAGS.bn_locations)
    bn_locations = get_bn_config_train(model_name=FLAGS.model_name, bn_int=FLAGS.bn_locations)
    FLAGS.load_pretrained = set_load_pretrained(FLAGS.train, FLAGS.test_run)

    # define test run params
    if FLAGS.test_run: index, bn_string, test_acc, adv_accs, result_log = set_test_run()
     

    # dict selection based on mode
    if FLAGS.save_to_log:
            columns_csv = ['Run', 'Model', 'Dataset', 'Batch-Normalization', 
                           'Training Mode', 'Test Accuracy', 'Epsilon Budget']

    if FLAGS.capacity_regularization:
        FLAGS.epsilon_in = FLAGS.epsilon_in[0:4]
        if FLAGS.save_to_log:
            columns_csv = ['Run', 'Model', 'Dataset', 'Batch-Normalization', 
                           'Training Mode', 'beta-lagrange', 'Test Accuracy', 'Epsilon Budget']
    
    elif FLAGS.rank_init:
        FLAGS.epsilon_in = FLAGS.epsilon_in[0:3]
        if FLAGS.save_to_log:
            columns_csv = ['Run', 'Model', 'Dataset', 'Batch-Normalization', 
                           'Training Mode', 'pre-training-steps', 'Test Accuracy', 'Epsilon Budget']
       
    ################################################################################################

    # save to results log if file not already saved
    if FLAGS.save_to_log:
        csv_path = get_csv_path(FLAGS.model_name)
        print('CSV PATH: ', csv_path)
        if FLAGS.load_pretrained: 
            if FLAGS.test_frequency:
                already_exists = False
            else:
                already_exists = check_log(run_name=FLAGS.pretrained_name, log_file=csv_path)
            print('ALREAD EXISTS IN RESULTS LOG: ', already_exists)

    # carry out channel transfer only for full-BN configs
    if len(FLAGS.channel_transfer)>0:
        if get_bn_int_from_name(FLAGS.pretrained_name) not in [5]: 
            already_exists = True
    if FLAGS.capacity_calculation:
        if get_bn_int_from_name(FLAGS.pretrained_name)!= 100: 
            already_exists = True
    if FLAGS.frequency_analysis:
        if get_bn_int_from_name(FLAGS.pretrained_name) not in [100]: 
            already_exists = True
    if FLAGS.IB_noise_calculation:
        if get_bn_int_from_name(FLAGS.pretrained_name) not in [100]: 
            already_exists = True
    if FLAGS.parametric_frequency_MSE_CE or FLAGS.parametric_frequency_MSE:
        if get_bn_int_from_name(FLAGS.pretrained_name) not in [100]: 
            already_exists = True
    if FLAGS.compare_frequency_domain:
        if get_bn_int_from_name(FLAGS.pretrained_name)!= 100: 
            already_exists = True
    if FLAGS.adversarial_test and not FLAGS.train:
        if get_bn_int_from_name(FLAGS.pretrained_name) not in [0,100]:
            already_exists = True 
        elif get_bn_int_from_name(FLAGS.pretrained_name) == 0 and not FLAGS.use_pop_stats:
            already_exists = True 
    if FLAGS.adversarial_test and 'Square' in FLAGS.attacks_in :
        if FLAGS.use_pop_stats:
            if get_bn_int_from_name(FLAGS.pretrained_name) not in [100, 0]:
                already_exists = True 
        else:
            if get_bn_int_from_name(FLAGS.pretrained_name) not in [100]:
                already_exists = True 
    if FLAGS.adversarial_test and FLAGS.attenuate_HF:
        if get_bn_int_from_name(FLAGS.pretrained_name) != 100:
            already_exists = True
    '''if FLAGS.adversarial_test and 'PGD' in FLAGS.attacks_in:
        if FLAGS.dataset == 'SVHN' and FLAGS.use_pop_stats:
            already_exists = False '''
    if FLAGS.adversarial_transferrability:
        print('Model ATTACKING: ', FLAGS.pretrained_name)
        print('Model ATTACKED: ', FLAGS.pretrained_name_to_attack)
        if str(FLAGS.pretrained_name) == str(FLAGS.pretrained_name_to_attack):
            already_exists = True 
            print('The two models are the same')
        else:
            print('Good to go!')
    if FLAGS.test_low_pass_robustness:
        if get_bn_int_from_name(FLAGS.pretrained_name) not in [100]:
            already_exists = True
    if len(FLAGS.prune_mode) > 0 :
        if get_bn_int_from_name(FLAGS.pretrained_name) not in [100]:
            already_exists = True

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
        
        if FLAGS.test_frequency:
            test_acc = test(index, test_frequency=True)

        if FLAGS.IB_noise_calculation:
            _ = test(index, IB_noise_calculation=True)
        
        if FLAGS.parametric_frequency_MSE or FLAGS.parametric_frequency_MSE_CE:
            _ = test(index, parametric_frequency=True)
        
        if FLAGS.frequency_analysis:
            _ = test(index, frequency_analysis=True)

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
                if attack == 'PGD' or attack == 'FGSM':
                    for eps in FLAGS.epsilon_in:
                        FLAGS.epsilon = float(eps)
                        dict_name = attack + '-' + str(FLAGS.epsilon)
                        if FLAGS.capacity_calculation:
                            _ =  test(index, capacity_calculation=True)
                        elif FLAGS.attenuate_HF:
                            adv_accs[dict_name] = test(index, HF_attenuate=True)
                        else:
                            adv_accs[dict_name] = test(index, adversarial=True)
                        
                elif attack in ['FAB', 'APGD_CE', 'APGD_DLR', 'Square', '-PGD']:
                    if FLAGS.adversarial_test:
                        for eps in FLAGS.epsilon_in:
                            FLAGS.epsilon = float(eps)
                            dict_name = attack + '-' + str(FLAGS.epsilon)
                            if attack == 'Square':
                                adv_accs[dict_name] = test(index, square_attack=True)
                            else:
                                adv_accs[dict_name] = get_FAB_acc(index, attack)
                                
        if FLAGS.test_low_pass_robustness:
            adv_accs = dict()
            for attack in FLAGS.attacks_in:
                FLAGS.attack = attack
                if attack == 'PGD':
                    for eps in FLAGS.epsilon_in:
                        FLAGS.epsilon = float(eps)
                        FLAGS.radii_to_test = [16, 15, 14, 13, 12, 11, 10]
                        for radius in FLAGS.radii_to_test:
                            FLAGS.low_pass_radius = int(radius)
                            dict_name = attack + '-' + str(FLAGS.epsilon) + '-' + str(FLAGS.low_pass_radius)
                            adv_accs[dict_name] = test(index, test_low_pass_robustness=True)
        
        if len(FLAGS.prune_mode) > 0 :
            adv_accs = dict()
            prune_percentages = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            for prune_percentage in prune_percentages:
                FLAGS.prune_percentage = prune_percentage
                dict_name = 'percentage - ' + str(prune_percentage)
                adv_accs[dict_name] = test(index, standard=True)

        if FLAGS.compare_frequency_domain:
            _ = test(index, compare_frequency_domain=True) 
        
        if FLAGS.adversarial_transferrability:
            FLAGS.epsilon = float(FLAGS.epsilon)
            transfer_acc = test(index, adversarial_transferrability=True)

            # get dict name of where it is stored in memory
            transfer_dict_name = 'trasfer_' + str(FLAGS.attacks_in[0]) + '_' + str(FLAGS.epsilon).replace('.', '') + '.npy'
            counter_dict_name = 'counter_' + str(FLAGS.attacks_in[0]) + '_' + str(FLAGS.epsilon).replace('.', '') + '.npy'

            # get name of key and sub-key
            key_name_attacker = str(FLAGS.pretrained_name.split('_')[0] + '_' + FLAGS.pretrained_name.split('_')[1])
            key_name_attacked = str(FLAGS.pretrained_name_to_attack.split('_')[0] + '_' + FLAGS.pretrained_name_to_attack.split('_')[1])

            if str(os.getcwd()).find('bitbucket') != -1:
                if FLAGS.dataset=='SVHN':
                    root_to_save = './gpucluster/SVHN/'
                elif FLAGS.dataset=='CIFAR10':
                    root_to_save = './gpucluster/CIFAR10/'
                elif FLAGS.dataset=='CIFAR100':
                    root_to_save = './gpucluster/CIFAR100/'
            else:
                root_to_save = './results/'

            # create and save dict (if it doesn't exist yet)
            if not os.path.isfile(root_to_save + 'adversarial_transferrability/' + transfer_dict_name):

                counter_dict = {key_name_attacker: {key_name_attacked: 1}}
                np.save(root_to_save + 'adversarial_transferrability/' + counter_dict_name, counter_dict)

                transfer_dict = {key_name_attacker: {key_name_attacked: transfer_acc}}
                np.save(root_to_save + 'adversarial_transferrability/' + transfer_dict_name, transfer_dict)

                print(transfer_dict)

            # update and save dict (if it already exists)
            else:
                counter_dict = np.load(root_to_save + 'adversarial_transferrability/' + counter_dict_name, allow_pickle='TRUE').item()
                transfer_dict = np.load(root_to_save + 'adversarial_transferrability/' + transfer_dict_name, allow_pickle='TRUE').item()

                print(transfer_dict)

                if key_name_attacker in list(transfer_dict.keys()):
                    print('KEY NAME ATTACKER **ALREADY** IN DICT')
                    if key_name_attacked in list(transfer_dict[key_name_attacker].keys()):
                        print('KEY NAME **ATTACKED** **ALREADY** IN DICT')

                        counter_dict[key_name_attacker][key_name_attacked] += 1
                        
                        curr_mean = transfer_dict[key_name_attacker][key_name_attacked]
                        updated_count = counter_dict[key_name_attacker][key_name_attacked]

                        print('COUNT: ', updated_count)

                        updated_mean = (((updated_count-1)/updated_count)*curr_mean) + (transfer_acc/updated_count)
                        transfer_dict[key_name_attacker][key_name_attacked] = updated_mean

                    else:
                        counter_dict[key_name_attacker][key_name_attacked] = 1
                        transfer_dict[key_name_attacker][key_name_attacked] = transfer_acc
                else:
                    print('KEY NAME ATTACKER **NOT YET** IN DICT')
                    to_update_counter = {key_name_attacker: {key_name_attacked: 1}}
                    counter_dict.update(to_update_counter)

                    to_update_transfer = {key_name_attacker: {key_name_attacked: transfer_acc}}
                    transfer_dict.update(to_update_transfer)
                
                print(transfer_dict)

                # save dict
                np.save(root_to_save + 'adversarial_transferrability/' + transfer_dict_name, transfer_dict)
                np.save(root_to_save + 'adversarial_transferrability/' + counter_dict_name, counter_dict)

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

                elif FLAGS.rank_init:
                    csv_dict = {
                        columns_csv[0] : index,
                        columns_csv[1] : model_name_,
                        columns_csv[2] : FLAGS.dataset,
                        columns_csv[3] : bn_string, 
                        columns_csv[4] : FLAGS.mode,
                        columns_csv[5] : FLAGS.pre_training_steps,
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
                # fill csv dictionaruìy based on different modes of training/testing   
                csv_dict = dict()
                for i, log in enumerate(result_log):
                    if i <=5:
                        csv_dict[columns_csv[i]] = log
                if FLAGS.test:
                    csv_dict[columns_csv[5]] = test_acc
                elif FLAGS.test_frequency:
                    print(test_acc)
                    freq_dict = {'radius': int(FLAGS.frequency_radius), \
                                 'frequency_accuracy': float(test_acc)}
                    # csv_dict[columns_csv[6]] = int(FLAGS.frequency_radius)
                    # csv_dict[columns_csv[7]] = test_acc
                    csv_dict.update(freq_dict) 
                elif len(FLAGS.prune_mode) > 0:
                    pass
                elif not FLAGS.adversarial_test and not FLAGS.test_low_pass_robustness: 
                    adv_accs = {}

                csv_dict.update(adv_accs)    
            
            elif len(result_log)>1 and (FLAGS.capacity_regularization or FLAGS.rank_init):
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
