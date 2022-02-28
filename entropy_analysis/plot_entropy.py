import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from absl import app

files = os.listdir('./entropy_analysis/entropies/')

for layers_entropy in files:
    H_ls = np.load('./entropy_analysis/entropies/' + layers_entropy)

    means = np.zeros((H_ls.shape[0], ))
    stds = np.zeros((H_ls.shape[0], ))

    for layer_num in range(H_ls.shape[0]):
        print(H_ls[layer_num, :])
        means[layer_num] = np.mean(H_ls[layer_num, :])
        stds[layer_num] = np.std(H_ls[layer_num, :])
    
    fig = plt.figure()
    plt.errorbar(np.arange(0, H_ls.shape[0]), means, yerr=stds, fmt='.k')
    plt.show()
    name_out = layers_entropy.replace('.npy', '.jpg')
    plt.savefig('./entropy_analysis/' + name_out)

'''
def main(argv):
    
    del argv

if __name__ == '__main__':
    app.run(main)
'''