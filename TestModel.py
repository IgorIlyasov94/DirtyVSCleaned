import numpy as np
import pandas as pd

import os
import shutil

from tqdm import tqdm

import torch
import torch.utils.data
import torch.nn.functional as functional
import torchvision
import matplotlib.pyplot as plt
import time
import copy

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
        def __getitem__(self, index):
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            path = self.imgs[index][0]
            tuple_with_path = (original_tuple + (path,))
            return tuple_with_path

def test_model(val_transforms, batch_size, model, device):
    test_dir = 'test'

    shutil.copytree(os.path.join(data_root, 'test'), os.path.join(test_dir, 'unknown'))

    test_dataset = ImageFolderWithPaths('/kaggle/working/test', val_transforms)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()

    test_predictions = []
    test_img_paths = []

    for inputs, labels, paths in tqdm(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            preds = model(inputs)

        test_predictions.append(functional.softmax(preds, dim=1)[:,1].data.cpu().numpy())
        test_img_paths.append(paths)

    test_predictions = np.concatenate(test_predictions)

    submission_df = pd.DataFrame.from_dict({'id': test_img_paths, 'label': test_predictions})

    submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred > 0.5 else 'cleaned')
    submission_df['id'] = submission_df['id'].str.replace('/kaggle/working/test/unknown/', '')
    submission_df['id'] = submission_df['id'].str.replace('.jpg', '')

    submission_df.set_index('id', inplace=True)
    submission_df.head(n=6)

    submission_df.to_csv('submission.csv')