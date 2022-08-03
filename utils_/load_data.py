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

    def __init__(self, frequency_radius=15, p=0.5):
        super().__init__()
        assert(isinstance(frequency_radius, int))

        self.frequency_radius = frequency_radius
        self.p = p
    
    # numpy version
    ###################################################################
    '''def fft(self, img):
        return np.fft.fft2(img)

    def fftshift(self, img):
        return np.fft.fftshift(self.fft(img))

    def ifft(self, img):
        return np.fft.ifft2(img)

    def ifftshift(self, img):
        return self.ifft(np.fft.ifftshift(img))'''
    
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
        print('--------------------------------: ', Images.shape)
        if len(Images.shape) == 4:
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

        elif len(Images.shape) == 3:
            mask = self.mask_radial(np.zeros([Images.shape[0], Images.shape[1]]), r)

            tmp_low = np.zeros([Images.shape[0], Images.shape[1], Images.shape[2]])
            for j in range(Images.shape[2]):
                fd = self.fftshift(Images[:, :, j])
                fd = fd * mask
                img_low = self.ifftshift(fd)
                tmp_low[:,:,j] = np.real(img_low)

            tmp_high = np.zeros([Images.shape[0], Images.shape[1], Images.shape[2]])
            for j in range(Images.shape[2]):
                fd = self.fftshift(Images[:, :, j])
                fd = fd * (1 - mask)
                img_high = self.ifftshift(fd)
                tmp_high[:,:,j] = np.real(img_high)

        return tmp_low, tmp_high
    ###################################################################
    
    # torch version
    ###################################################################
    def fft(self, img):
        return torch.fft.fft2(img)

    def fftshift(self, img):
        return torch.fft.fftshift(self.fft(img))

    def ifft(self, img):
        return torch.fft.ifft2(img)

    def ifftshift(self, img):
        return self.ifft(torch.fft.ifftshift(img))

    def distance(self, i, j, imageSize, r):
        dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
        if dis < r:
            return 1.0
        else:
            return 0
    
    def mask_radial_torch(self, img, r):
        rows, cols = img.size()
        mask = torch.zeros(rows, cols)
        for i in range(rows):
            for j in range(cols):
                mask[i, j] = self.distance(i, j, imageSize=rows, r=r)
        return mask

    def get_low_frequecny_img(self, Images, r):
        # get mask
        #print('__________________________________________', type(Images))
        if len(Images.size()) == 4:
            mask = self.mask_radial(torch.zeros(Images.size(2), Images.size(3)), r)
            mask = mask.view(1, 1, mask.size(0), mask.size(1)).expand(Images.size())

        elif len(Images.size()) == 3:
            mask = self.mask_radial_torch(torch.zeros(Images.size(1), Images.size(2)), r)
            #print('.............................................',mask.size())
            mask = mask.view(1, mask.size(0), mask.size(1)).expand(Images.size())

        # carry out FT
        fc = self.fftshift(Images) 
        
        # apply mask 
        fc_masked = fc * mask

        # carry out IFT
        rec_img = torch.real(self.ifftshift(fc_masked))

        return rec_img
    ###################################################################

    def forward(self, img):
        if torch.rand(1) < self.p:
            if isinstance(img, torch.Tensor) or isinstance(img, torch.tensor):
                low_img = self.get_low_frequecny_img(img, self.frequency_radius)
                return low_img
            else:
                low_img, _ = self.generateDataWithDifferentFrequencies_3Channel(np.array(img), self.frequency_radius)
                return Image.fromarray((low_img * 255).astype(np.uint8))
        else:
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

    elif FLAGS.train_with_low_frequency:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=5, sigma=(1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            LowPass(),
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
    if FLAGS.adversarial_test and 'Square' in FLAGS.attacks_in :
        FLAGS.batch_size = 32
    test_loader = DataLoader(test_set, batch_size=FLAGS.batch_size, shuffle=False)

    return train_loader, test_loader