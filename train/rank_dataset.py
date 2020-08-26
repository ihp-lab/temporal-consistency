import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision.transforms import ToTensor, Scale, Compose, Pad, RandomHorizontalFlip, CenterCrop, RandomCrop, Scale, ToPILImage 
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import os, sys
import torch
from tqdm import tqdm

image_base_folder = '../vox2_crop_fps25'

class CustomDatasetFromImages(Dataset):
    def __init__(self, transformations, spacing):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """

        self.seed = np.random.seed(567)
        self.transform = transformations
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.video_label = sorted(os.listdir(image_base_folder))
        frame_index = []
        self.video_label_suff = []

        length = spacing * 10 + 1

        for video in tqdm(self.video_label):
            path = image_base_folder + '/' + video
            frames = sorted(os.listdir(path), key=lambda x: int(x[:-4]))

            if len(frames) - length> 0:
                index = np.random.choice(range(len(frames) - length), size=1)
                frame_index.append(index)
                self.video_label_suff.append(video)


        self.frame_index = frame_index
        # Calculate len
        self.data_len = len(self.video_label_suff)
        self.spacing = spacing


    def __getitem__(self, index):
        # Get image name from the pandas df
        video_off = int(index % 1)
        video_base_index = int((index - video_off) / 1)
        anchor_index = self.frame_index[video_base_index][video_off]
        pos_index = anchor_index + 1

        path = image_base_folder + '/' + self.video_label_suff[video_base_index]
        frames = sorted(os.listdir(path), key=lambda x: int(x[:-4]))
        random_frame = frames[anchor_index]
        close_frame = frames[pos_index]

        far_frame_name_list = []

        for i in range(1,11):
            neg_index = i * self.spacing + anchor_index + 1
            far_frame = frames[neg_index]
            far_name = path + '/' + far_frame
            far_frame_name_list.append(far_name)

        source_name = path + '/' + random_frame
        close_name = path + '/' + close_frame
        
        # Open image
        try:
            s_img = Image.open(source_name)
            c_img = Image.open(close_name)


            f_imgs = []
            for far in far_frame_name_list:
                f_img = Image.open(far)
                f_imgs.append(f_img)
     
        except FileNotFoundError:
            print("sample missing use first")
            return self.__getitem__(0)

        imgs = [0] * 12
        img_s = self.transform(s_img)
        imgs[0] = img_s

        img_c = self.transform(c_img)
        imgs[1] = img_c

        for i in range(10):
            img_f = self.transform(f_imgs[i])
            imgs[2+i] = img_f
        single_image_label = self.video_label_suff[video_base_index]

        return (imgs, single_image_label)

    def __len__(self):
        return self.data_len
