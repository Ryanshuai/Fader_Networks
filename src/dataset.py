import os
import torch
import torch.utils.data as data
import cv2
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, params, train_eval = 'train', one_hot = Tr):
        # 一条example 为 图像文件名和对应的属性
        # 读取样本的图像文件名和属性

        # path
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', train_eval)
        img_path = os.path.join(data_path, 'img')
        attr_file = open(os.path.join(data_path, 'attributes.txt'))

        # readline
        self.examples_nums = int(attr_file.readline()) # 第一行是样本数
        attr_all_names =  [name for name in attr_file.readline().split()] # 第二行是属性名，共40个
        examples = []
        for line in attr_file.readlines():
            line_content = [cnt for cnt in line.split()]  #  例 [000001.jpg, 1, -1, 1, ...] size : 41
            one_img_filename = img_path + "/" + line_content[0]  # data/train/img/000001.jpg
            one_img_attrs = [] # 一副图像的所有属性，如Smiling和Male属性
            for name, _ in params.attr:
                if one_hot:
                    if float(line_content[attr_all_names.index(name) + 1]) == 1.:
                        one_img_attrs.append([0,1])
                    else:
                        one_img_attrs.append([1,0])
                else:
                    if float(line_content[attr_all_names.index(name) + 1]) == 1.:
                        one_img_attrs += [1]
                    else:
                        one_img_attrs += [0]
            examples.append((one_img_filename, np.array(one_img_attrs)))  # [(data/train/img/000001.jpg,[0.,1.,1.,0.]), ((data/train/img/000002.jpg,[0.,1.,1.,0.])),...]
        self.examples = examples
        self.train = True if train_eval == 'train' else False
        self.param = params

    def transform(self, img):
        # resize
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

        # data augmentation, flip
        if self.train:
            if self.param.v_flip and np.random.rand() <= 0.5:
                img = cv2.flip(img, 0)
            if self.param.h_flip and np.random.rand() <= 0.5:
                img = cv2.flip(img, 0)
        img = torch.from_numpy(img)

        # normalization
        img = img.float().div_(255.0).mul_(2.0).add_(-1) #[-1, 1]
        img = img.permute(2,0,1) # [256,256,3] -> [3, 256,256]

        return img

    def __getitem__(self, index):
        img_filename, attr = self.examples[index]
        img = cv2.imread(img_filename)
        img = self.transform(img)
        return img, torch.FloatTensor(attr)

    def __len__(self):
        return self.examples_nums
