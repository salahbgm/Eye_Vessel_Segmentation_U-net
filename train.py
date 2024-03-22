import os
import time
from glob import glob 

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from model import build_unet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time

from data import DriveDataset

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x  = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss


def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()

    with torch.no_grad():    
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    # seed
    seeding(42)

    # directories
    create_dir("files")


    # dataloaders
    train_X = sorted(glob("data_2/train/images/*"))
    train_y = sorted(glob("data_2/train/masks/*"))

    valid_X = sorted(glob("data_2/test/images/*"))
    valid_y = sorted(glob("data_2/test/masks/*"))

    data_str = f"dataset size : \nTrain: {len(train_X)} - {len(train_y)}\nValid: {len(valid_X)} - {len(valid_y)}"
    print(data_str)


    # hyperparameters
    H = 512
    W = 512
    size = (H, W)
    batch_size = 16
    num_epochs = 250
    lr = 1e-4   
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")

    checkpoint_dir = "files/checkpoints.pth"
    

    # dataset and loader
    train_dataset = DriveDataset(train_X, train_y)
    valid_dataset = DriveDataset(valid_X, valid_y)


    train_loader = DataLoader(dataset = train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True,
                                num_workers=2
                              )

    valid_loader = DataLoader(dataset = valid_dataset, 
                              batch_size=batch_size, 
                              shuffle=False,
                                num_workers=2
                              )



    # create model
    device = torch.device('cuda')
    model = build_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.1, verbose=True)
    loss = DiceBCELoss()

    # training the model

    best_valid_loss = float('inf')
    print("0")

    for epoch in range(num_epochs):
        start_time = time.time()
        print("1")

        training_loss = train(model, train_loader, optimizer, loss, device)
        valid_loss = train(model, valid_loader, optimizer, loss, device)
        print("2")

        # save mdoel
        if valid_loss < best_valid_loss:
            print("3")

            data_str = f"Validation loss decreased from {best_valid_loss:.4f} to {valid_loss:.4f}. Saving the model to {checkpoint_dir}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_dir)
        print("4")

        end_time = time.time()  
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print("5")

        data_str = f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s"
        data_str += f"\n\tTrain Loss: {training_loss:.3f}"
        data_str += f"\n\t Val. Loss: {valid_loss:.3f}"
        print("6")
        print(data_str)