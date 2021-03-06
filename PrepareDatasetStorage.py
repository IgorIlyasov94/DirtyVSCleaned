import numpy as np
import pandas as pd

import os
import shutil

from tqdm import tqdm

def prepare_dataset_storage(data_root, train_dir, val_dir, test_dir, class_names):
    for dir_name in [train_dir, val_dir]:
        for class_name in class_names:
            os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)

    for class_name in class_names:
        source_dir = os.path.join(data_root, train_dir, class_name)
        for i, file_name in enumerate(tqdm(os.listdir(source_dir))):
            if i % 6 != 0:
                dest_dir = os.path.join(train_dir, class_name)
            else:
                dest_dir = os.path.join(val_dir, class_name)
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))

    shutil.copytree(os.path.join(data_root, 'test'), os.path.join(test_dir, 'unknown'))
