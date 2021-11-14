import matplotlib.pyplot as plt
import numpy as np
import os

from absl import app
from absl import flags
from utils import utils_flags
from utils.load_data import FLAGS

def main(argv):
    
    del argv
    FLAGS = flags.FLAGS

    model_names = os.listdir('./logs/adv/')

    for model_name in model_names:

        path = './logs/adv/' + str(model_name) + '/'
        model_runs = os.listdir(path)

        fig = plt.figure()
        for i, filename in enumerate(model_runs):
            
            if filename.split('.')[1] != 'npy':
                continue 

            arr_1 = np.load(path+filename)
            c = ['r', 'b', 'g', 'y']
            plt.plot(FLAGS.epsilon, arr_1.reshape(19, 1), c[i])
        
        plt.show()
        plt.savefig(path + model_name + '.png')

if __name__ == '__main__':
    app.run(main)