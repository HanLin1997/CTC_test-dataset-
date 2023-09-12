import torch
from torch.utils.data import Dataset
import glob
import re
import os
import cv2
import numpy as np
from dictionary import *

def collate_fn(batch):
    maxw,imgs,maxl,labels,length = 0,[],0,[],[]
    for item in batch:
        img,label= item
        maxw = max(img.shape[1],maxw)
        imgs.append(img)
        maxl = max(len(label),maxl)

        labels.append(label)
        length.append(len(label) - 1)
    
    maxw = max(maxw,46)
    maxw = maxw + 16 - (maxw - 30)%16   #insure maxw = 16k + 30, k is integer. easy to caculate final length

    #padding same width in batch
    iw = []
    for i in range(len(imgs)):
        h, w = imgs[i].shape
        iw.append(w)
        pw = maxw - w
        if pw > 0:
            imgs[i] = np.pad(imgs[i],((0,0),(0,pw)),mode='constant',constant_values=255.0)
        
    for i,l in enumerate(length):
        pl = maxl - l
        if pl > 0:
            labels[i] = np.pad(labels[i],(0,pl),mode='constant',constant_values=word_dict['#'])
    
    imgs = torch.tensor(np.array(imgs)).unsqueeze(1)
    lbs = torch.tensor(np.array(labels))
    lls = torch.tensor(np.array(length))

    return imgs,lbs,lls


class MyDataset(Dataset):
    def __init__(self):
        self.items = []
        self.maxlength = 0
        fs = glob.glob('./image/*.txt')
        for fn in fs:
            with open(fn, 'r') as f:
                lines = f.readlines()
            for line in lines:
                pair = line.strip().split(',')
                if len(pair) != 2:
                    continue
                path = pair[0].strip()
                label = pair[1].strip()
                self.maxlength = max(len(label), self.maxlength)
                self.items.append( (f"./image/{path}",self.label_str2seq(label)) )
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        path,label= self.items[index]
        if not os.path.exists(path):
            print(path)
            raise(f'path {path} not exists.')
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img, label

    def label_str2seq(self, str):
        str = re.sub('\s+',' ', str)
        label = []
        for c in str:
            if c in word_dict:
                label.append(word_dict[c])
            else:
                label.append(word_dict['*'])
        label.append(word_dict['$'])
        return label


if __name__ == '__main__':
    dataset = MyDataset()
    len(dataset)
    print(dataset[0])