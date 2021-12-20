import pickle
import os
import os.path as op
import csv
import argparse


def get_image_list(test_img_path):
    image_list = os.listdir(test_img_path)
    img_name_list = []
    for img_name in image_list:
        img_name_list.append(img_name)
    return img_name_list

def get_seed_list():
    path = "data/train"
    file_list = os.listdir(path)
    seed_list = []
    for file in file_list:
        seed_list.append(file)
    return seed_list

def get_seed(index):
    seed_list = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
                 'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
    key = int(index)
    return seed_list[key]

def run_test(test_data_path, model):
    test_data = pickle.load(open(test_data_path, 'rb'))
    pre = model.predict(test_data)
    return pre

def write_result_in_csv(result_path, img_name_list, result):
    write = []
    for i in range(len(img_name_list)):
        write.append([img_name_list[i], get_seed(result[i])])
    with open(result_path, 'w', newline='') as f:
        csv_file = csv.writer(f)
        csv_file.writerow(['file', 'species'])
        csv_file.writerows(write)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_path",
        default="result/submission_hog_base_1.csv",
        type=str,
        help="is train or not.",
    )
    parser.add_argument(
        "--test_img_path",
        default="data/test",
        type=str,
        help="The file of test data.",
    )
    parser.add_argument(
        "--test_data_path",
        default="data/pre_data/test_data_hog_base.pkl",
        type=str,
        help="The vector of test data.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="checkpoints/model_hog_base_1.pkl",
        type=str,
        help="saved model. ",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_arguments()
    img_list = get_image_list(args.test_img_path)
    model = pickle.load(open(args.checkpoint_path, 'rb'))
    result = run_test(args.test_data_path, model)
    write_result_in_csv(args.result_path, img_list, result)
    print("save successfully!")

