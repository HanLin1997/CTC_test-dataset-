import torch
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import torch.utils.data
from model import Net
from dataset import MyDataset, collate_fn

batch_size = 8
max_img_length = 520
max_label_length = 25
dictionary = {'#': 0, '^': 1, '$': 2, '*': 3,   # '#': blank, '^': start, '$': end
              '0': 4, '1': 5, '2': 6, '3': 7, '4': 8, '5': 9, '6': 10, '7': 11, '8': 12, '9': 13,
              'a': 14, 'b': 15, 'c': 16, 'd': 17, 'e': 18, 'f': 19, 'g': 20, 'h': 21,
              'i': 22, 'j': 23, 'k': 24, 'l': 25, 'm': 26, 'n': 27, 'o': 28, 'p': 29,
              'q': 30, 'r': 31, 's': 32, 't': 33, 'u': 34, 'v': 35, 'w': 36, 'x': 37,
              'y': 38, 'z': 39, 'A': 40, 'B': 41, 'C': 42, 'D': 43, 'E': 44, 'F': 45,
              'G': 46, 'H': 47, 'I': 48, 'J': 49, 'K': 50, 'L': 51, 'M': 52, 'N': 53,
              'O': 54, 'P': 55, 'Q': 56, 'R': 57, 'S': 58, 'T': 59, 'U': 60, 'V': 61,
              'W': 62, 'X': 63, 'Y': 64, 'Z': 65}

def read_data(data_size):
    #read image

    data_train = []
    for i in range(data_size):
        image = cv2.imread(f"./image/{i}.jpg", cv2.IMREAD_GRAYSCALE)
        #image = torch.tensor(image)
        if image.shape[1] < max_img_length:
            #padding = np.ones((50, max_length - image.shape[1]),dtype=np.float64)*255
            #padding = torch.tensor(padding)
            #image = torch.cat((image, padding), 1)
            pad_w = max_img_length - image.shape[1]
            image = np.pad(image, ((0,0),(0,pad_w)), mode='constant', constant_values=255)
        image = torch.tensor(image)
        image = torch.unsqueeze(image, 0)
        data_train.append(image)
    data_train = torch.stack(data_train, 0)
    #data_train = torch.tensor(data_train, dtype=torch.float32).unsqueeze(1)
    
    #read label
    label_train = np.array([])
    with open("./image/label.txt", 'r') as f:
        str_label_list = f.readlines()
    target_lengths = list(map(lambda x:len(x.strip().split(',')[1]), str_label_list))
    target_lengths = torch.tensor(np.array(target_lengths))
    for str_label in str_label_list:
        str_label = str_label.strip().split(',')[1]
        padding_length = max_label_length - len(str_label) - 1
        str_label = str_label + '$' + '#'*padding_length
        lable = list(map(lambda x:dictionary[x], str_label))
        label_train = np.append(label_train, np.array(lable))
    label_train = torch.tensor(label_train)
    label_train = label_train.view(-1, max_label_length)
    if data_train.shape[0] == label_train.shape[0]:
        return data_train, label_train, target_lengths

def data_lorder(data):
    train_dataset = torch.utils.data.TensorDataset(data[0], data[1], data[2])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 drop_last=True)
    return train_loader


def train(net, train_loader, optimizer):
    for batch_id, (data,target,target_lengths) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data.float())
        #output = output.view(30, batch_size, 66)
        output = output.squeeze(2).permute(2, 0, 1)
        input_lenghs = torch.full((batch_size,), output.shape[0], dtype=torch.long)
        loss = F.ctc_loss(output, target, input_lenghs, target_lengths, zero_infinity=True)
        loss.backward()
        optimizer.step()
        print(f"loss = {loss.item()}      [{batch_id*len(data)}/{len(train_loader.dataset)}]")

dataset = MyDataset()
train_loader = torch.utils.data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 drop_last=True,
                                 sampler=None, 
                                 batch_sampler=None, 
                                 collate_fn=collate_fn)
#data = read_data(1000)
#train_loader = data_lorder(data)
net = Net()
net.float()
optimizer = optim.AdamW(net.parameters(), lr=0.01)
epoch = 0
while epoch < 8:
    epoch = epoch + 1
    print(f"epoch: {epoch}")
    train(net, train_loader, optimizer)