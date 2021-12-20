from data_utils import get_train_data, device_data_label, save_pickle
from sklearn import svm
from xgboost import XGBClassifier
import random
import pickle
from sklearn.neighbors import KNeighborsClassifier
import argparse


def train(arg):
    train_data = get_train_data(arg.train_data_path)
    random.shuffle(train_data)
    train, eval = train_data[:3800], train_data[3800:]
    X_train, y_train = device_data_label(train)
    X_test, y_test = device_data_label(eval)
    # model = XGBClassifier(max_depth=15,
    #                       learning_rate=0.05,
    #                       n_estimators=2000,
    #                       min_child_weight=4,
    #                       max_delta_step=0,
    #                       subsample=0.7,
    #                       colsample_bytree=0.7,
    #                       reg_alpha=0,
    #                       reg_lambda=0.4,
    #                       objective='binary:logistic',
    #                       missing=None,
    #                       eval_metric='auc',
    #                       seed=1440,
    #                       gamma=0
    #                       )
    # cfl = svm.SVC()
    # cfl = KNeighborsClassifier(n_neighbors=8)
    model = XGBClassifier()
    # model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    print("train done!")
    pre = model.predict(X_test)
    num = 0
    for i in range(len(y_test)):
        if y_test[i] == pre[i]:
            num += 1
    print("准确率："+str(num / len(pre)))
    save_pickle(arg.save_checkpoint_path, model)


def eval(arg):

    model = pickle.load(open(arg.checkpoint_path, 'rb'))
    train_data = get_train_data(arg.train_data_path)
    random.shuffle(train_data)
    train, eval = train_data[:3800], train_data[3800:]
    X_test, y_test = device_data_label(eval)
    pre = model.predict(X_test)
    num = 0
    for i in range(len(y_test)):
        if y_test[i] == pre[i]:
            num += 1
    print("准确率："+str(num / len(pre)))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--is_train",
        default=True,
        type=bool,
        help="is train or not.",
    )
    parser.add_argument(
        "--train_data_path",
        default="data/pre_data/train_data_hog_base.pkl",
        type=str,
        help="The file of train data.",
    )
    parser.add_argument(
        "--save_checkpoint_path",
        default="checkpoints/model_hog_base_1.pkl",
        type=str,
        help="The file dir to save checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="checkpoints/model_hog.pkl",
        type=str,
        help="saved model. ",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arguments()
    if not args.is_train:
        eval(args)
    else:
        train(args)
