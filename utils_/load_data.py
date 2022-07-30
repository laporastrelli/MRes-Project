import os 
import numpy as np
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from absl import flags
from utils_ import utils_flags
from PIL import Image


FLAGS = flags.FLAGS


class LowPass(torch.nn.Module):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, frequency_radius=15, p=0.5):
        assert(isinstance(frequency_radius, int))
        self.frequency_radius = frequency_radius
        self.p = p
    
    def fft(self, img):
        return np.fft.fft2(img)

    def fftshift(self, img):
        return np.fft.fftshift(self.fft(img))

    def ifft(self, img):
        return np.fft.ifft2(img)

    def ifftshift(self, img):
        return self.ifft(np.fft.ifftshift(img))
    
    def distance(self, i, j, imageSize, r):
        dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
        if dis < r:
            return 1.0
        else:
            return 0

    def mask_radial(self, img, r):
        rows, cols = img.shape
        mask = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                mask[i, j] = self.distance(i, j, imageSize=rows, r=r)
        return mask
    
    def generateDataWithDifferentFrequencies_3Channel(self, Images, r):
        Images_freq_low = []
        Images_freq_high = []
        mask = self.mask_radial(np.zeros([Images.shape[1], Images.shape[2]]), r)
        for i in range(Images.shape[0]):
            tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
            for j in range(3):
                fd = self.fftshift(Images[i, :, :, j])
                fd = fd * mask
                img_low = self.ifftshift(fd)
                tmp[:,:,j] = np.real(img_low)
            Images_freq_low.append(tmp)
            tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
            for j in range(3):
                fd = self.fftshift(Images[i, :, :, j])
                fd = fd * (1 - mask)
                img_high = self.ifftshift(fd)
                tmp[:,:,j] = np.real(img_high)
            Images_freq_high.append(tmp)

        return np.array(Images_freq_low), np.array(Images_freq_high)

    def forward(self, img):
        if torch.rand(1) < self.p:
            low_image, _ = self.generateDataWithDifferentFrequencies_3Channel(np.array(img), self.frequency_radius)
            return Image.fromarray(low_image)
        return img


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
    
    if FLAGS.train_with_GaussianBlurr:
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=5, sigma=(1.0)),
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
    

    train_loader = DataLoader(train_set, batch_size=FLAGS.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=FLAGS.batch_size, shuffle=False)

    return train_loader, test_loader