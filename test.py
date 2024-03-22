import os 
import time
from operator import add
import numpy as np 
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import jaccard_score, accuracy_score, f1_score, precision_score, recall_score

from model import build_unet
from utils import create_dir, seeding


def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_true = y_true >0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred >0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    #jaccard_score
    jaccard = jaccard_score(y_true, y_pred)

    #accuracy
    accuracy = accuracy_score(y_true, y_pred)

    #f1 score
    f1 = f1_score(y_true, y_pred)

    #precision
    precision = precision_score(y_true, y_pred)

    #recall
    recall = recall_score(y_true, y_pred)

    return jaccard, accuracy, f1, precision, recall


def mask_parse(mask):
   mask = np.expand_dims(mask, axis=-1) # (512,512,1)
   mask = np.concatenate([mask, mask, mask], axis=-1) # (512,512,3)
   return mask

if __name__ == "__main__":
    # seed
    seeding(42)

    # directories
    create_dir("files")

    #load dataset

    test_X = sorted(glob("data_2/test/images/*"))[:20]
    test_y = sorted(glob("data_2/test/masks/*"))[:20]

    #hyperparameters
    H = 512
    W = 512
    size = (H, W)
    checkpoint = "files/checkpoints.pth"

    #load the checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x,y) in tqdm(enumerate(zip(test_X, test_y)), total=len(test_X)):
        name= x.split("/")[-1].split(".")[0]
        

        #read image
        image = cv2.imread(x, cv2.IMREAD_COLOR) # (512,512,3)
        #image = cv2.resize(image, size)
        x = np.transpose(image, (2,0,1)) # (3,512,512)
        x = x/255.0
        x = np.expand_dims(x, axis=0) # (1,3,512,512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)
        

        #read mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE) # (512,512)
        #mask = cv2.resize(mask, size)
        y = np.expand_dims(mask, axis=0) # (1,512,512)
        y = y/255.0
        y = np.expand_dims(y, axis=0) # (1,1,512,512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)

        #prediction
        with torch.no_grad():
            start_time = time.time()
            pred_y = model(x)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))
            pred_y = pred_y[0].cpu().numpy()
            pred_y = np.squeeze(pred_y, axis = 0)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        # calculate average metrics
        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate([image, line, ori_mask, line, pred_y*255], axis=1)
        cv2.imwrite(f"files/{name}.png", cat_images)
    
    jaccard_score = metrics_score[0]/len(test_X)
    f1_score = metrics_score[1]/len(test_X)
    recall_score = metrics_score[2]/len(test_X) 
    precision_score = metrics_score[3]/len(test_X)
    accuracy_score = metrics_score[4]/len(test_X)
    print(f"Jaccard Score: {jaccard_score}", f"F1 Score: {f1_score}", f"Recall Score: {recall_score}", f"Precision Score: {precision_score}", f"Accuracy Score: {accuracy_score}")

    fps = 1/np.mean(time_taken)
    print(f"FPS: {fps}")
    


