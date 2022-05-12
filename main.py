############ IMPORTS ############
from webbrowser import get
import os
import numpy as np
import csv
import torch
from torch.utils.tensorboard import SummaryWriter
from utils_ import utils_flags
from absl import app
from absl import flags
from functions.train import train
from functions.test import test
from get_FAB_acc import get_FAB_acc
from utils_.get_model import get_model
from utils_.load_data import get_data
from utils_.test_utils import adversarial_test
from utils_.miscellaneous import get_epsilon_budget, get_bn_int_from_name, get_bn_config_train, set_load_pretrained

# TODO: change this to be adaptive to the number of attacks and epsilon used
columns_csv = ['Run', 'Model', 'Dataset', 'Batch-Normalization', 
               'Training Mode', 'Test Accuracy', 'Epsilon Budget',
               'PGD - 0.1', 'PGD - 0.0313', 'PGD - 0.5', 'PGD - 0.1565']

def main(argv):
    
    del argv

    ######################################################### SETUP #########################################################

    # parse inputs 
    FLAGS = flags.FLAGS
    
    if FLAGS.device is None:
        # get device 
        FLAGS.device = 'cuda:' + str(torch.cuda.current_device())

    # retrive dataset-corresponding epsilon budget
    FLAGS.epsilon_in = get_epsilon_budget(dataset=FLAGS.dataset)

    # model name logistics
    if FLAGS.model_name.find('ResNet50_v') != -1:
        FLAGS.model_name = 'ResNet50'        

    # get BN locations from pretrained model name
    if FLAGS.load_pretrained:
        result_log = FLAGS.result_log.split(',')
        FLAGS.bn_locations = get_bn_int_from_name(run_name=FLAGS.pretrained_name)
        print('Run name: ', FLAGS.pretrained_name)
        print('BN integer:', FLAGS.bn_locations)
    
    # get model name, based on it determine one-hot encoded BN locations 
    model_name = FLAGS.model_name
    bn_locations = get_bn_config_train(model_name=FLAGS.model_name, bn_int=FLAGS.bn_locations)
    FLAGS.load_pretrained = set_load_pretrained(FLAGS.train, FLAGS.test_run)
        
    # define test run params
    if FLAGS.test_run:
        FLAGS.train = False
        FLAGS.test = False
        FLAGS.adversarial_test = False
        FLAGS.load_pretrained = False
        FLAGS.save_to_log = True
        
        index = 'try'  
        FLAGS.model_name = 'test'
        FLAGS.dataset = 'test dataset'
        bn_string = 'test bn'
        FLAGS.mode = 'test mode'   
        test_acc = 10000000000
        FLAGS.epsilon_in = [10000000, 77777777]
        adv_accs = {'PDG-0.1': 99999999999,
                    'PGD-0.2': -0.11111111, 
                    'PGD-0.3': -0.444444}
        
        result_log = [index, FLAGS.model_name, FLAGS.dataset, bn_string, FLAGS.mode, test_acc]
        FLAGS.csv_path = './results/test.csv'

    print('Model Name: ', model_name)
    print('Dataset: ', FLAGS.dataset)
    print('Epsilon Budget: ', FLAGS.epsilon_in) 
    print('BN Configuration: ', bn_locations)
    
    ######################################################### OPERATIONS #########################################################

    where_bn = bn_locations
    
    if FLAGS.train:
        if not FLAGS.test_run:
            index = train(model_name, where_bn)
            result_log=[]

    elif FLAGS.load_pretrained:
        index = FLAGS.pretrained_name

    if FLAGS.test:
        test_acc = test(index)

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

    if FLAGS.adversarial_test:
        adv_accs = dict()
        for attack in FLAGS.attacks_in:
            FLAGS.attack = attack
            if attack == 'PGD':
                for eps in FLAGS.epsilon_in:
                    FLAGS.epsilon = float(eps)
                    dict_name = attack + '-' + str(FLAGS.epsilon)
                    adv_accs[dict_name] = test(index, adversarial=True)
                    
            if attack in ['FAB', 'APGD-CE', 'APGD_DLR', 'Square', '-PGD']:
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
            csv_dict = {
                columns_csv[0] : index,
                columns_csv[1] : model_name_,
                columns_csv[2] : FLAGS.dataset,
                columns_csv[3] : bn_string, 
                columns_csv[4] : FLAGS.mode, 
                columns_csv[5] : test_acc,
                columns_csv[6] : FLAGS.epsilon_in}
            csv_dict.update(adv_accs)

        elif len(result_log)>1:    
            csv_dict = dict()
            for i, log in enumerate(result_log):
                if i <=5:
                    csv_dict[columns_csv[i]] = log
            if FLAGS.test:
                csv_dict[columns_csv[5]] = test_acc
            csv_dict.update(adv_accs)    
            
        try:
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


