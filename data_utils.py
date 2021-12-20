import os
import os.path as op
import json
import cv2
import base64
import numpy as np
import torch
import torch.nn.functional as F
import PIL
import matplotlib.pyplot as plt
import pickle


def create_img(img):
    f_img = np.array(img, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(f_img)
    gray = 2 * g - b - r
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    # bin_img = bin_img / 255
    return bin_img


def get_vector(img):
    bin_img = create_img(img)
    height_vector = np.sum(bin_img, axis=1)
    width_vector = np.sum(bin_img, axis=0)
    # height_vector.sort()
    # width_vector.sort()
    return height_vector, width_vector


def cut_down(h, w):
    # height = torch.tensor(h)
    # height = F.softmax(height, dim=0)
    # height = np.array(height)
    # height = height[:200]
    height = h[:400]
    # weight = torch.tensor(w)
    # weight = F.softmax(weight, dim=0)
    # weight = np.array(weight)
    # weight = weight[:200]
    weight = w[:400]
    vector = np.concatenate((height, weight), axis=0)
    if len(vector) < 800:
        type_ids = np.zeros(800 - len(vector))
        vec = np.concatenate((vector, type_ids), axis=0)
    else:
        vec = vector
    return vec


def create_label(path):
    label = {}
    num = 0
    file_list = os.listdir(path)
    for file in file_list:
        label[file] = float(num)
        num += 1
    return label

def get_label(label, file_name):
    label_arr = np.array([label[file_name]])
    return label_arr


def save_pickle(save_path, data):
    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle)

def get_train_data(data_path):
    pickle_file = open(data_path, 'rb')
    data = pickle.load(pickle_file)
    return data

def device_data_label(data):
    x = []
    y = []
    for i in data:
        temp = i[1:].tolist()
        x.append(temp)
        y.append(i[0])
    return x, y


def average_filter(img, block_size, stride):
    result = []
    epoch = int(img.shape[0]/stride)
    for i in range(epoch):
        h = i * stride
        for j in range(epoch):
            w = j * stride
            block = img[h:h+block_size][w:w+block_size]
            result.append(block.sum())
    res = np.array(result)
    return res