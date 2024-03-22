import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip,VerticalFlip, Rotate


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    train_X = sorted(glob(os.path.join(path, "train", "output_images" , "*.png")))
    train_y = sorted(glob(os.path.join(path, "train","output_masks" , "*.png")))

    test_X = sorted(glob(os.path.join(path, "test", "output_images" , "*.png")))
    test_y = sorted(glob(os.path.join(path, "test", "output_masks" , "*.png")))

    return (train_X, train_y), (test_X, test_y)

def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)   

    for idx,(x,y) in tqdm(enumerate (zip(images, masks)), total = len(images)):
        name = x.split("\\")[-1].split(".")[0]

        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y= imageio.mimread(y)[0]

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image = x, mask = y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image = x, mask = y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit = 45, p=1.0)
            augmented = aug(image = x, mask = y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]


        else : 
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip (X,Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "images", tmp_image_name)
            mask_path = os.path.join(save_path, "masks", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)


            index += 1


        


if __name__ == "__main__":
    np.random.seed(42)


    data_path = "D:/unetv2/data_"
    (train_X, train_y), (test_X, test_y) = load_data(data_path)

    print(f"train_X: {len(train_X)} - train_y: {len(train_y)}")
    print(f"test_X: {len(test_X)} - test_y: {len(test_y)}")

    create_dir("D:/unetv2/data_2/train/images/")
    create_dir("D:/unetv2/data_2/train/masks/")
    create_dir("D:/unetv2/data_2/test/images/")
    create_dir("D:/unetv2/data_2/test/masks/")

    augment_data(train_X, train_y, "D:/unetv2/data_2/train/", augment=True)
    augment_data(test_X, test_y, "D:/unetv2/data_2/test/", augment=False)

