# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import random,cv2
from PIL import Image
import glob,os
from numpy.testing._private.utils import decorate_methods
# from torch.utils.data import Dataset

import kornia
import numpy as np
import os,glob
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as transforms
from mindspore.dataset.transforms.py_transforms import Compose
# from mindspore.dataset import ImageFolderDataset
from mindspore import Tensor
from mindspore import dtype as mstype
# import torch
# from torchvision import transforms

#获取4个下采样对应的pip install opencv-contrib-pythonh
def get_H(im1,im2): #cv2.imread+RGB
    # im1 = cv2.imread('/home/sharklet/database/aftercut/train/left/2009.png')
    # im2 = cv2.imread('/home/sharklet/database/aftercut/train/right/2009.png')
    # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)  # (H,W,3)
    # im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    H_list = []
    for resize_scale in [1]: #[1,2,4,8]: #[2, 4, 8, 16]:
        # print(im1.shape)
        # resize
        resize_im1 = cv2.resize(im1, (im1.shape[1] // resize_scale, im1.shape[0] // resize_scale))  # W,H
        resize_im2 = cv2.resize(im2, (im2.shape[1] // resize_scale, im2.shape[0] // resize_scale))
        #
        surf = cv2.xfeatures2d.SURF_create()
       
        kp1, des1 = surf.detectAndCompute(resize_im1, None)
        kp2, des2 = surf.detectAndCompute(resize_im2, None)
        # 匹配特征点描述子
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        # 提取匹配较好的特征点
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        # 通过特征点坐标计算单应性矩阵H
        # （findHomography中使用了RANSAC算法剔初错误匹配）
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # 获取H后，但要放进tensor中的变换
        try:
            h = Tensor(H,mstype.float32)
            # h = torch.from_numpy(H.astype(np.float32))  # 否则float64，与网络中的tensor不匹配！ torch.from_numpy的作用是将numpy转化为torch张量
        except:
            h = None
        #     print(resize_scale)
        # h_inv = torch.inverse(h) #求逆
        H_list.append(h)
    return H_list

# class ImageFolder(Dataset):
class ImageFolder():
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories: ::
        - rootdir/
            - train/
                -left/
                    - 0.png
                    - 1.png
                -right/
            - test/
                -left/
                    - 0.png
                    - 1.png
                -right/
    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """
    def __init__(self, root, transform=None,patch_size=(256,256), split='train',need_file_name = False):
        splitdir = Path(root) / split  # 相当于osp.join

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        splitdir_left = splitdir / "left"
        splitdir_right = splitdir / "right"

        self.left_list = sorted(glob.glob(os.path.join(splitdir_left,"*")))
        self.right_list = sorted(glob.glob(os.path.join(splitdir_right, "*")))

        self.patch_size = patch_size
        #只保留了ToTensor
        self.transform = transform

        self.need_file_name = need_file_name

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        # img1 = Image.open(self.left_list[index]).convert('RGB')
        # img2 = Image.open(self.right_list[index]).convert('RGB')
        if os.path.basename(self.left_list[index]) != os.path.basename(self.right_list[index]):
            print(self.left_list[index])
            raise ValueError("cannot compare pictures.")
        ##
        img1 = cv2.imread(self.left_list[index])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(self.right_list[index])
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1 = np.ascontiguousarray(img1)   #之前报错，上网查到需要用这个方法
        img2 = np.ascontiguousarray(img2)
        #random cut for pair
        H, W, _ = img1.shape

        #randint是闭区间
        # print(H)
        # print(W)
        # print(self.patch_size)
        # print(H)
        # raise ValueError("stopxx")
        if self.patch_size[0]==H:
            startH = 0
            startW = 0
        else:
            startH = random.randint(0,H-self.patch_size[0]-1)
            startW = random.randint(0,W-self.patch_size[1]-1)

        img1 = img1[startH:(startH + self.patch_size[0]), startW:(startW + self.patch_size[1])]
        img2 = img2[startH:(startH + self.patch_size[0]), startW:(startW + self.patch_size[1])]
        # cv2.imshow('img1',img1)
        # raise ValueError("stop")
        
        H_list = get_H(img1,img2)
        
        ##
        if H_list[0]==None:
            print(self.left_list[index])
            print(self.right_list[index])
            #raise ValueError("None!!H_matrix")
            # 只有ToTensor
            if self.transform:
                return self.transform(img1), self.transform(img2) # ,H_list[1],H_list[2],H_list[3]
            return img1, img2  # ,H_list[1],H_list[2],H_list[3]

        #只有ToTensor
        if self.transform:
            # return self.transform(img1),self.transform(img2),H_list[0] #,H_list[1],H_list[2],H_list[3]
            if self.need_file_name:
                return self.transform(img1), self.transform(img2), H_list[0], os.path.basename(self.left_list[index])  # ,H_list[1],H_list[2],H_list[3]
            else:
                return self.transform(img1), self.transform(img2), H_list[0]  # ,H_list[1],H_list[2],H_list[3]

        if self.need_file_name:
            return img1, img2, H_list[0],os.path.basename(self.left_list[index])  # ,H_list[1],H_list[2],H_list[3]
        else:
            return img1,img2,H_list[0] #,H_list[1],H_list[2],H_list[3]

    def __len__(self):
        return len(self.left_list)


if __name__ == '__main__':
    train_transforms = Compose([
        transforms.ToPIL(), #需要先将numpy array转换为PIL image
        transforms.RandomCrop(128,128),
        transforms.ToTensor()
    ])
    train_dataset = ImageFolder('E:/Files/Onedrive/OneDrive - buaa.edu.cn/Scsx/aftercut',
                                split='train',
                                transform=train_transforms, patch_size = (128,128))
    print(train_dataset.__getitem__(0))