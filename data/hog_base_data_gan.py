import os
import os.path as op
import cv2
import numpy as np
import PIL
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_utils import get_vector, cut_down, create_label, get_label, save_pickle, create_img, average_filter
import torch
import torch.nn.functional as F
import pickle

train_data = []
# data_path = "data/train/Maize/"
path = "train"
save_path = "pre_data/train_data_hog_base.pkl"
file_list = os.listdir(path)
label_dict = create_label(path)
# img_list = os.listdir(data_path)
hog = cv2.HOGDescriptor()
for file in file_list:
    data_path = op.join(path, file)
    img_list = os.listdir(data_path)
    label = get_label(label_dict, file)
    for img_p in tqdm(img_list):
        img_key = img_p.split(".")[0]
        img_path = op.join(data_path, img_p)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        image = create_img(img)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        vector = hog.compute(image, winStride=(64, 64), padding=(0, 0))
        imgs = cv2.resize(img, (1000, 1000))
        h, w = get_vector(imgs)
        feature_2 = np.concatenate((h, w), axis=0)
        feature_1 = np.concatenate((vector, feature_2), axis=0)
        # print(len(vector))
        feature = np.concatenate((label, feature_1), axis=0)
        train_data.append(feature)


save_pickle(save_path, train_data)