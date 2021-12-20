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
save_path = "pre_data/train_data_plus.pkl"
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
        img = cv2.resize(img, (400, 400))
        # print(len(feature))
        feature_1 = average_filter(img, 20, 20)
        # t_feature = np.concatenate((label, feature), axis=0)
        # train_data.append(t_feature)
        h, w = get_vector(img)
        feature_2 = cut_down(h, w)
        feature_3 = cut_down(w, h)
        feature = np.concatenate((feature_2, feature_1), axis=0)
        t_feature = np.concatenate((label, feature), axis=0)
        train_data.append(t_feature)
        f = np.concatenate((feature_3, feature_1), axis=0)
        h_feature = np.concatenate((label, feature), axis=0)
        train_data.append(h_feature)

save_pickle(save_path, train_data)