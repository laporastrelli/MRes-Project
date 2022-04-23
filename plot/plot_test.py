import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


name = 'ln'
# epsilons = [2/255, 5/255, 8/255, 10/255, 12/255, 16/255, 0.1, 0.2]
epsilons = [2/255, 5/255, 8/255, 10/255, 12/255, 16/255, 0.1]
modes = []
results = os.listdir('./plot/' + name + '/')

fig = plt.figure()
i = 0
for j, result in enumerate(results):
    print(result)
    if result.find('.npy') != -1:
        if result.split('_')[1] == '0':
            modes.append('no-BN')
        elif result.split('_')[1] == '100':
            modes.append('full-BN')
        elif result.split('_')[1] == '1':
            modes.append('first block-BN')
        elif result.split('_')[1] == '2':
            modes.append('second block-BN')   
        elif result.split('_')[1] == '3':
            modes.append('third block-BN')
        elif result.split('_')[1] == '4':
            modes.append('fourth block-BN')
        elif result.split('_')[1] == '5':
            modes.append('fifth block-BN')
            
        to_plot = np.load('./plot/' + name + '/' + result)
        if to_plot.shape[0] != 7:
            to_plot = to_plot[1]
        plt.plot(epsilons, to_plot)
        plt.ylabel('Accuracy')
        plt.xlabel('Epsilon Budget')
        i+=1
        plt.show()

print(modes)
plt.legend(modes)
plt.savefig('./plot/' + name + '/' + 'test.jpg')