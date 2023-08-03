from os.path import join
from torchvision.transforms import Compose, ToTensor
from dataset import DatasetFromFolderEval, DatasetFromFullEval, DatasetFromHdf5, DatasetFromFullHdf5

def transform():
    return Compose([
        ToTensor(),
    ])

def get_training_set(gt_dir, panl_dir, lrms_dir, pan_dir):
    return DatasetFromHdf5(gt_dir, panl_dir, lrms_dir, pan_dir, transform=transform())

def get_fulltraining_set(data_dir):
    return DatasetFromFullHdf5(data_dir, transform=transform())

def get_test_set(gt_dir, panl_dir, lrms_dir, pan_dir):
    return DatasetFromHdf5(gt_dir, panl_dir, lrms_dir, pan_dir, transform=transform())

def get_fulltest_set(lrms_dir, pan_dir):
    return DatasetFromFullEval(lrms_dir, pan_dir, transform=transform())


