from utils_ import utils_flags
from absl import app
from absl import flags

def set_test_run():
    
    # parse inputs 
    FLAGS = flags.FLAGS

    FLAGS.train = False
    FLAGS.test = False
    FLAGS.adversarial_test = False
    FLAGS.load_pretrained = False
    FLAGS.save_to_log = True
    FLAGS.plot = False
    
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

    # print(index, bn_string, test_acc, adv_accs, result_log)

    return index, bn_string, test_acc, adv_accs, result_log