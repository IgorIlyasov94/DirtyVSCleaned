import torch
import torch.utils.data
import numpy as np
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
import time
import copy

from tqdm import tqdm

import PrepareDatasetStorage
import TransformDataset
import ShowDataset
import TrainModel
import TestModel

if __name__ == '__main__':
    data_root = 'plates'

    train_dir = 'train'
    val_dir = 'val'
    test_dir = 'test'

    class_names = ['cleaned', 'dirty']

    #PrepareDatasetStorage.prepare_dataset_storage(data_root, train_dir, val_dir, test_dir, class_names)

    train_dataset = TransformDataset.transform_dataset(train_dir, TransformDataset.train_transforms)
    val_dataset = TransformDataset.transform_dataset(val_dir, TransformDataset.val_transforms)

    batch_size = 8

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    #ShowDataset.show_dataset(val_dataloader, class_names)

    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


    TrainModel.train_model(model, loss, optimizer, scheduler, 100, device, train_dataloader, val_dataloader)


    TestModel.test_model(TransformDataset.val_transforms, batch_size, model, device)