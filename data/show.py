import os
import os.path as op
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_utils import get_vector, cut_down, create_label, get_label, save_pickle, create_img, average_filter
import torch
import torch.nn.functional as F
import pickle

train_data = []
# data_path = "data/train/Maize/"
path = "train"
save_path = "pre_data/train_data.pkl"
file_list = os.listdir(path)
label_dict = create_label(path)
# img_list = os.listdir(data_path)
hog = cv2.HOGDescriptor()

data_path = op.join(path, 'Blackgrass')
img_list = os.listdir(data_path)
# label = get_label(label_dict, file)
for img_p in tqdm(img_list):
    img_key = img_p.split(".")[0]
    img_path = op.join(data_path, img_p)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (400, 400))
    img = create_img(img)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    break