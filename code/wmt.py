import torch
from preprocess import get_language_model_data

def get_wmt_data():
    AUTOGRADER_TRAIN_FILE = '../data/wmt_train'
    AUTOGRADER_TEST_FILE = '../data/wmt_test'

    train_loader = get_language_model_data(AUTOGRADER_TRAIN_FILE)
    test_loader = get_language_model_data(AUTOGRADER_TEST_FILE)

    return train_loader, test_loader