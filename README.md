# Northeast University machine learning course assignment

## Task description

Plant Seedlings Classification in Kaggle

https://www.kaggle.com/c/plant-seedlings-classification/overview

## Train

command

```python
python train.py --is_train True --train_data_path data/pre_data/train_data_hog_base.pkl --save_checkpoint_path checkpoints/model_hog_base_1.pkl --checkpoint_path checkpoints/model_hog.pkl
```

## Test

command

```python
python test.py --result_path result/submission_hog_base_1.csv --test_img_path data/test --test_data_path data/pre_data/test_data_hog_base.pkl --checkpoint_path checkpoints/model_hog_base_1.pkl
```

