############ IMPORTS ############
import pandas as pd
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import utils_flags
from absl import app
from absl import flags
from functions.train import train
from functions.test import test
from utils.get_model import get_model

columns = ['Model', 'Dataset', 'Batch-Normalization', 
          'Training Mode', 'Test Accuracy', 'Epsilon Budget',
          'FGSM Test Accuracy', 'PGD Test Accuracy']

def main(argv):
    
    del argv

    FLAGS = flags.FLAGS

    # get inputs 
    model_names = [FLAGS.model_name]
    bn_locations = [int(i) for i in FLAGS.bn_locations]
    print(bn_locations)
    if FLAGS.train:
        FLAGS.load_pretrained == False
    elif FLAGS.train and not FLAGS.test_run:
        FLAGS.load_pretrained == True

    print(FLAGS.model_name, 
          FLAGS.train, 
          FLAGS.load_pretrained, 
          FLAGS.pretrained_name, 
          FLAGS.bn_locations)

    # define test run params
    if FLAGS.test_run:
        model_names = ['']
        FLAGS.train = False
        FLAGS.test = False
        FLAGS.adversarial_test = False
        FLAGS.load_pretrained = False

        FLAGS.model_name = 'test'
        FLAGS.dataset = 'test dataset'
        FLAGS.mode = ''   
        bn_string = 'try_1'
        FLAGS.epsilon = 7
        test_acc = 0
        adv_acc_FGSM = 0.1
        adv_acc_PGD = 0.2

        index = 'try'  

    # create results log file if it does not exist
    if not os.path.isfile('./logs/results.pkl') and FLAGS.save_to_log:
        df = pd.DataFrame(columns=columns)
        df.to_pickle('./logs/results.pkl')

    # open pandas results log file
    if FLAGS.save_to_log:
        df = pd.read_pickle('./logs/results.pkl')

    # attacks to be used
    attacks = ['FGSM', 'PGD']

    # first train, test (both standard and adversarial)
    for model_name in model_names:

        # define where to insert bact-norm layers (ResNet)
        if model_name.find('ResNet')!= -1:
            assert len(bn_locations) == 4, "BatchNorm locations for ResNet models should be equal to 4."

        # define where to insert bact-norm layers (VGG)
        elif model_name.find('VGG')!= -1:
            assert len(bn_locations) == 5, "BatchNorm locations for VGG models should be equal to 5."

        # for each bn location train and test the model                    
        for where_bn in [bn_locations]:

            # create dictonaries to be inserted into Pandas Dataframe
            if sum(where_bn)==0:
                bn_string = 'No'
            elif sum(where_bn)>1:
                bn_string = 'Yes - ' + 'all'
            else:
                bn_string = 'Yes - ' + str(where_bn.index(1) + 1) + ' of ' + str(len(where_bn))
            print(bn_string)
            
            # train and test the model and get results
            if FLAGS.train:
                index = train(model_name, where_bn)

            elif FLAGS.load_pretrained:
                index = FLAGS.pretrained_name

            if FLAGS.test:
                test_acc = test(index)

            for attack in attacks:
                
                # set attack to FLAGS so it can used in other files
                FLAGS.attack = attack

                # carry out adversarial training as well as cross-BN adversarial transferability test
                if FLAGS.adversarial_test:
                    adv_acc = test(index, adversarial=True)
                
                    if attack == 'FGSM':
                        adv_acc_FGSM = adv_acc
                    if attack == 'PGD':
                        adv_acc_PGD = adv_acc

            # create dictionary for results log
            if FLAGS.save_to_log:
                # dict
                df_dict = {
                columns[0] : FLAGS.model_name,
                columns[1] : FLAGS.dataset,
                columns[2] : bn_string, 
                columns[3] : FLAGS.mode, 
                columns[4] : test_acc,
                columns[5] : FLAGS.epsilon, 
                columns[6] : adv_acc_FGSM, 
                columns[7] : adv_acc_PGD, 
                }

                to_df = []
                for df_el in df_dict:
                    to_df.append(df_dict[df_el])

                df_to_add = pd.DataFrame(np.array(to_df).reshape(1, len(to_df)), columns=columns, index=[index])
                df = df.append(df_to_add)
                df.to_pickle('./logs/results.pkl')
                df.to_csv('./logs/results.csv')

if __name__ == '__main__':
    app.run(main)



'''bn_locations = [[1,1,1,1,1],
                   [0,0,0,0,0],
                   [1,0,0,0,0],
                   [0,1,0,0,0],
                   [0,0,1,0,0],
                   [0,0,0,0,1]]'''

'''bn_locations = [[1,1,1,1],
                  [0,0,0,0],
                  [1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]]'''