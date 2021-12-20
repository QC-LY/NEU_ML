import os
import os.path as op
import cv2
import numpy as np
import PIL
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_utils import get_vector, cut_down, create_label, get_label, save_pickle, create_img, average_filter

hog = cv2.HOGDescriptor()
test_data = []
path = "test"
save_path = "pre_data/test_data_plus.pkl"
image_list = os.listdir(path)
for image in tqdm(image_list):
    img_key = image.split(".")[0]
    img_path = op.join(path, image)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (400, 400))
    feature_1 = average_filter(img, 20, 20)
    h, w = get_vector(img)
    feature_2 = cut_down(h, w)
    feature = np.concatenate((feature_2, feature_1), axis=0)
    test_data.append(feature)
    # image = create_img(img)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # vector = hog.compute(image, winStride=(16, 16), padding=(0, 0))
    # # print(len(vector))
    test_data.append(feature)

save_pickle(save_path, test_data)