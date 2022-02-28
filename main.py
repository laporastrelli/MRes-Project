############ IMPORTS ############
from webbrowser import get
import pandas as pd
import os
import numpy as np
import csv
from torch.utils.tensorboard import SummaryWriter
from utils_ import utils_flags
from absl import app
from absl import flags
from functions.train import train
from functions.test import test
from functions.get_DBA_acc import get_DBA_acc
from get_FAB_acc import get_FAB_acc
from utils_.get_model import get_model
from utils_.load_data import get_data
from utils_.test_utils import adversarial_test

# TODO: change this to be adaptive to the number of attacks and epsilon used
columns_csv = ['Run', 'Model', 'Dataset', 'Batch-Normalization', 
               'Training Mode', 'Test Accuracy', 'Epsilon Budget',
               'PGD - 0.1', 'PGD - 0.0313', 'PGD - 0.5', 'PGD - 0.1565']

root_columns_csv = ['Run', 'Model', 'Dataset', 'Batch-Normalization', 
                    'Training Mode', 'Test Accuracy']


def main(argv):
    
    del argv

    ######################################################### SETUP #########################################################

    # parse inputs 
    FLAGS = flags.FLAGS

    # epsilon budget for CIFAR10
    if FLAGS.dataset == 'CIFAR10':
        # 2/255, 5/255, 8/255, 10/255, 12/255, 16/255, 0.1, 0.2
        FLAGS.epsilon_in = [0.0392, 0.0980, 0.1565, 0.1961, 0.2352, 0.3137, 0.5, 1,]
    
    # epsilon budget for SVHN
    if FLAGS.dataset == 'SVHN':
        # 2/255, 5/255, 8/255, 0.1, 0.2
        FLAGS.epsilon_in = [0.0157, 0.0392, 0.0626, 0.2, 0.4]    
    
    # model name logistics
    if FLAGS.model_name.find('ResNet50_v') != -1:
        FLAGS.model_name = 'ResNet50'        

    # get BN locations from pretrained model name
    if FLAGS.load_pretrained:
        print(FLAGS.result_log)
        result_log = FLAGS.result_log.split(',')
        temp = FLAGS.pretrained_name.split('_')[1]
        if temp == 'bn':
            FLAGS.bn_locations = 100
        elif temp == 'no':
            FLAGS.bn_locations = 0
        else:
            # add 1 for consistency with name 
            FLAGS.bn_locations = int(temp) + 1

    # get model name, based on it determine one-hot encoded BN locations 
    model_name = FLAGS.model_name
    if FLAGS.model_name.find('VGG') != -1:
        if FLAGS.bn_locations==100:
            bn_locations = [1,1,1,1,1]
        elif FLAGS.bn_locations==0:
            bn_locations = [0,0,0,0,0]
        else:
            bn_locations = [i*0 for i in range(5)]
            bn_locations[int(FLAGS.bn_locations-1)] = 1
            print(bn_locations)
    elif FLAGS.model_name.find('ResNet')!= -1:
        if FLAGS.bn_locations==100:
            bn_locations = [1,1,1,1]
        elif FLAGS.bn_locations==0:
            bn_locations = [0,0,0,0]
        else:
            bn_locations = [i*0 for i in range(4)]
            bn_locations[int(FLAGS.bn_locations-1)] = 1

    # get modes (eg train, test, adversarial test)
    if FLAGS.train:
        FLAGS.load_pretrained = False
    elif not FLAGS.train and not FLAGS.test_run:
        FLAGS.load_pretrained = True
        
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

    # display run modes
    print(FLAGS.model_name, 
          FLAGS.train, 
          FLAGS.load_pretrained, 
          FLAGS.pretrained_name, 
          bn_locations)

    print('BATCH SIZE: ', FLAGS.batch_size)
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
                for eps in FLAGS.epsilon_in:
                    FLAGS.epsilon = float(eps)
                    _ = test(index, get_features=True, adversarial=True)
        else:
            _ = test(index, get_features=True)
        
        # setting adversarial_test back to False:
        FLAGS.adversarial_test = False

    if FLAGS.adversarial_test:
        adv_accs = dict()
        for attack in FLAGS.attacks_in:
            FLAGS.attack = attack
            if attack != 'DBA' and attack != 'FAB':
                for eps in FLAGS.epsilon_in:
                    FLAGS.epsilon = float(eps)
                    dict_name = attack + '-' + str(FLAGS.epsilon)
                    adv_accs[dict_name] = test(index, adversarial=True)
            else:
                dict_name = attack
                adv_accs[dict_name] = test(index, adversarial=True)

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
        
        name = FLAGS.model_name + '_' + bn_string
        to_save = np.asarray(result_log[6:]).astype(np.float64)
        path_out = './plot/' + name + '/' + name +  '_' + FLAGS.which + '.npy'

        print(name)

        if os.path.isfile(path_out):
            temp = np.load(path_out)
            out = np.vstack((temp, to_save))
        else:
            out = to_save

        np.save(path_out, out)

if __name__ == '__main__':
    app.run(main)










'''
            eps_list = [str(result_log[6]), str(result_log[7])] + FLAGS.epsilon_in

            idx = idx+1
            csv_dict[columns_csv[idx]] = eps_list
            
            idx = idx+1
            csv_dict[columns_csv[idx]] = result_log[9]

            idx = idx+1
            csv_dict[columns_csv[idx]] = result_log[10]



elif attack == 'DBA':
                adv_acc_DBA = get_DBA_acc(index)
            elif attack =='PGD':
                pgd_dict = dict()
                for eps in FLAGS.epsilon_in:
                    FLAGS.epsilon = float(eps)
                    adv_acc = test(index, adversarial=True)
                    pgd_dict[str(eps)] = adv_acc
            elif attack == 'FAB':
                for eps in FLAGS.epsilon_in:
                    FLAGS.epsilon = float(eps)
                    adv_acc_FAB, mean_dist = get_FAB_acc(index)
'''

'''
# create results log file if it does not exist
if not os.path.isfile('./logs/results.pkl') and FLAGS.save_to_log:
    df = pd.DataFrame(columns=columns)
    df.to_pickle('./logs/results.pkl')

# open pandas results log file
if FLAGS.save_to_log:
    df = pd.read_pickle('./logs/results.pkl')
    
columns = ['Model', 'Dataset', 'Batch-Normalization', 
          'Training Mode', 'Test Accuracy', 'Epsilon Budget',
          'FGSM Test Accuracy', 'PGD Test Accuracy', 'DBA Test Accuracy']
# dict
df_dict = {
    columns[0] : model_name_,
    columns[1] : FLAGS.dataset,
    columns[2] : bn_string, 
    columns[3] : FLAGS.mode, 
    columns[4] : test_acc,
    columns[5] : FLAGS.epsilon, 
    columns[6] : adv_acc_FGSM, 
    columns[7] : adv_acc_PGD, 
    columns[8] : adv_acc_DBA
}

to_df = []
for df_el in df_dict:
    to_df.append(df_dict[df_el])

df_to_add = pd.DataFrame(np.array(to_df).reshape(1, len(to_df)), columns=columns, index=[index])
df = df.append(df_to_add)
df.to_pickle('./logs/results.pkl')
df.to_csv('./logs/results.csv')
'''