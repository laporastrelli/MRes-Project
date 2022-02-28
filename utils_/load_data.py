import os 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from absl import flags
from utils_ import utils_flags

FLAGS = flags.FLAGS

def get_data():

    # dataset download decision
    download = False

    if FLAGS.mode == 'optimum':
        FLAGS.batch_size = 400

    # performs transforms on data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    # prevents from downloading it if dataset already downloaded
    if not os.path.exists(FLAGS.dataset_path + FLAGS.dataset):
        os.mkdir(FLAGS.dataset_path + FLAGS.dataset)
        download = True
    else:
        if len(os.listdir(FLAGS.dataset_path + FLAGS.dataset)) == 0:
            download = True

    if FLAGS.dataset == 'CIFAR10':
        train_set = datasets.CIFAR10(FLAGS.dataset_path + FLAGS.dataset, train=True, download=download, transform=transform_train)
        test_set = datasets.CIFAR10(FLAGS.dataset_path + FLAGS.dataset, train=False, download=download, transform=transform_test)

    elif FLAGS.dataset == 'SVHN':
        train_set = datasets.SVHN(FLAGS.dataset_path + FLAGS.dataset, 
                                  split='train', 
                                  transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                  download=download)

        test_set = datasets.SVHN(FLAGS.dataset_path + FLAGS.dataset, 
                                 split='test', 
                                 transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                 download=download)
    
    # create loaders
    ## it is important NOT to shuffle the test dataset since the adversarial variation 
    ## delta are going to be saved in memory in the same order as the test samples are. 
    
    print(type(test_set))

    train_loader = DataLoader(train_set, batch_size=FLAGS.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=FLAGS.batch_size, shuffle=False)

    return train_loader, test_loader