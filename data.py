from os import listdir
from os.path import join
from PIL import Image
from os.path import basename
import torch.utils.data as data

def is_image_file(filename):
  filename_lower = filename.lower()
  return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

class DatasetFromFolder_2(data.Dataset):
    def __init__(self, image_dir, transform=None):
        super(DatasetFromFolder_2, self).__init__()
        data_dir = "%s/data/" % image_dir
        label_dir = "%s/label/" % image_dir

        self.data_filenames = []
        self.label_filenames = []
        for x in listdir(data_dir):
            self.data_filenames.append(join(data_dir, x))
            y = x.split('_')[0] + '.png'
            self.label_filenames.append(join(label_dir, y))

        self.transform = transform

    def __getitem__(self, index):

        data = Image.open(self.data_filenames[index])
        label = Image.open(self.label_filenames[index])

        if self.transform:
            data = self.transform(data)
            label = self.transform(label)

        return data, label

    def __len__(self):
        return len(self.data_filenames)

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, transform=None, layer=5):
        super(DatasetFromFolder, self).__init__()
        data_dir = "%s/data/" % image_dir
        label_dir = "%s/label/" % image_dir
        self.data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        self.label_filenames = [join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)]

        self.transform = transform
        self.layer = layer

    def __getitem__(self, index):
        if basename(self.data_filenames[index]) != basename(self.label_filenames[index]):
            print(self.data_filenames[index], self.label_filenames[index])
            raise ValueError('Name not equal')
        data = Image.open(self.data_filenames[index])
        label = Image.open(self.label_filenames[index])
        w, h = label.size
        label2 = label.resize((w//2, h//2), Image.BILINEAR)
        label3 = label.resize((w//4, h//4), Image.BILINEAR)
        label4 = label.resize((w//8, h//8), Image.BILINEAR)
        # if self.layer==5:
        label5 = label.resize((w//16, h//16), Image.BILINEAR)

        if self.transform:
            data = self.transform(data)
            label = self.transform(label)
            label2 = self.transform(label2)
            label3 = self.transform(label3)
            label4 = self.transform(label4)
            label5 = self.transform(label5)

        return data, label, label2, label3, label4, label5
            # , label5

    def __len__(self):
        return len(self.data_filenames)