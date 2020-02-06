
from os import listdir
from os.path import join
from PIL import Image
from os.path import basename
import torch.utils.data as Data
from torchvision import transforms as TF
import torch
import numpy as np
# from Utils.utils import get_mean_and_std
import re



def is_image_file(filename):
  filename_lower = filename.lower()
  return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

class derain_train_datasets(Data.Dataset):
    '''return rain_img ,clear , classfy_label'''
    def __init__(self, data_root):
        super(derain_train_datasets, self).__init__()
        self.data_filenames = [join(data_root, x) for x in listdir(data_root) if is_image_file(x) and '._' not in x]

        self.transform = TF.Compose(
                [
                    TF.ToTensor(),  # tensor range for 0 to 1
                ]
        )
    def __getitem__(self, index):
        data_path = self.data_filenames[index]
        data_img = Image.open(data_path)
        # data = self.transform(data)

        data1 = data_img.crop((0, 0, 512, 512))
        label = data_img.crop((512, 0, 1024, 512))
        w, h = label.size
        label2 = label.resize((w // 2, h // 2), Image.BILINEAR)
        label3 = label.resize((w // 4, h // 4), Image.BILINEAR)
        label4 = label.resize((w // 8, h // 8), Image.BILINEAR)
        label5 = label.resize((w // 16, h // 16), Image.BILINEAR)

        if self.transform:
            data1 = self.transform(data1)
            label = self.transform(label)
            label2 = self.transform(label2)
            label3 = self.transform(label3)
            label4 = self.transform(label4)
            label5 = self.transform(label5)

        return data1, label, label2, label3, label4, label5

    def __len__(self):
        return len(self.data_filenames)


class derain_test_datasets(Data.Dataset):
    '''return rain_img ,clear , classfy_label'''
    def __init__(self, data_root):
        super(derain_test_datasets, self).__init__()
        self.data_filenames = [join(data_root, x) for x in listdir(data_root) if is_image_file(x) and '._' not in x]

        self.transform = TF.Compose(
                [
                    TF.ToTensor(),  # tensor range for 0 to 1
                ]
        )
    def __getitem__(self, index):
        data_path = self.data_filenames[index]
        data = Image.open(data_path)
        data = self.transform(data)

        label = data[:, :, 512:1024]
        data1 = data[:, :, :512]

        return data1, label

    def __len__(self):
        return len(self.data_filenames)

class derain_test_datasets(Data.Dataset):
    '''return rain_img ,clear , classfy_label'''
    def __init__(self, data_root):
        super(derain_test_datasets, self).__init__()
        self.data_filenames = [join(data_root, x) for x in listdir(data_root) if is_image_file(x) and '._' not in x]

        self.transform = TF.Compose(
                [
                    TF.ToTensor(),  # tensor range for 0 to 1
                ]
        )
    def __getitem__(self, index):
        data_path = self.data_filenames[index]
        data = Image.open(data_path)
        data = self.transform(data)

        label = data[:, :, 512:1024]
        data1 = data[:, :, :512]

        return data1, label

    def __len__(self):
        return len(self.data_filenames)