

from data.ac17_dataloader import augment_gamma
from data.ac17_dataloader import AC17_2DLoad as load2D

import os, nibabel

from skimage import transform


# System libs
# Numerical libs
import torch
import torch.utils.data as data
from data.augmentations import Compose, RandomSizedCrop, AdjustContrast, AdjustBrightness, RandomVerticallyFlip, RandomHorizontallyFlip, RandomRotate, PaddingCenterCrop

import numpy as np
import cv2


class SideWalkData(data.Dataset):

    def __init__(self,
                 root,
                 split='train',
                 augmentations=None,
                 img_norm=True,
                 k=5,
                 k_split=1,
                 target_size=(256, 256)
                 ):
        self.target_size = target_size
        self.ROOT_PATH = root
        self.split = split
        self.k = k
        self.split_len = int(200/self.k)
        self.k_split = int(k_split)
        self.augmentations = augmentations
        self.TRAIN_IMG_PATH = os.path.join(root, 'train')
        self.TRAIN_SEG_PATH = os.path.join(root, 'train_seg')
        self.list = self.read_files()


    def read_files(self):

        l = []

        max_count = 100
        count = 0

        for root, dirs, files in os.walk(self.TRAIN_IMG_PATH, topdown=True):


            for f in files:

                if count >= 100:
                    break

                fname = f.split(".")[0]

                # path = root + "/" + f
                l.append(fname)

                count+=1
        return l

    def __len__(self):
        if self.split == "train":
            return 100 - self.split_len
        else:
            return self.split_len

    def __getitem__(self, i): # i is index
        filename = self.list[i]
        full_img_path = os.path.join(self.TRAIN_IMG_PATH, filename+".jpg")
        full_seg_path = os.path.join(self.TRAIN_SEG_PATH, filename+".png")

        img = cv2.imread(full_img_path)
        seg = cv2.imread(full_seg_path)



        if self.augmentations is not None:
            img = img.transpose(2, 0, 1)
            seg = seg.transpose(2, 0, 1)
            img_c = np.zeros((img.shape[0], self.target_size[0], self.target_size[1]))
            seg_c = np.zeros((seg.shape[0], self.target_size[0], self.target_size[1]))

            for z in range(img.shape[0]):
                if img[z].min() > 0:
                    img[z] -= img[z].min()

                img_tmp, seg_tmp = self.augmentations(img[z].astype(np.uint32), seg[z].astype(np.uint8))
                img_tmp = augment_gamma(img_tmp)

                mu = img_tmp.mean()
                sigma = img_tmp.std()
                img_tmp = (img_tmp - mu) / (sigma+1e-10)
                img_c[z] = img_tmp
                seg_c[z] = seg_tmp

            img = img_c.transpose(1,2,0)
            seg = seg_c.transpose(1,2,0)

        img = torch.from_numpy(img).float()
        seg = torch.from_numpy(seg).long()

        data_dict = {
            "name": filename,
            "image": img,
            "mask": seg,
        }

        return data_dict



class AC17Data(data.Dataset):

    def __init__(self,
                 root,
                 split='train',
                 augmentations=None,
                 img_norm=True,
                 k=5,
                 k_split=1,
                 target_size=(256, 256)
                 ):
        self.target_size = target_size
        self.ROOT_PATH = root
        self.split = split
        self.k = k
        self.split_len = int(200/self.k)
        self.k_split = int(k_split)
        self.augmentations = augmentations
        self.TRAIN_IMG_PATH = os.path.join(root)
        self.TRAIN_SEG_PATH = os.path.join(root)
        self.list = self.read_files()

    def read_files(self):
        d = []

        return d

    def __len__(self):
        if self.split == "train":
            return 200 - self.split_len
        else:
            return self.split_len

    def __getitem__(self, i): # i is index
        filename = "patient001_frame01"
        full_img_path = os.path.join(self.TRAIN_IMG_PATH, filename)
        full_seg_path = os.path.join(self.TRAIN_SEG_PATH, filename)
        img = nibabel.load(full_img_path+".nii.gz")
        seg = nibabel.load(full_seg_path+"_gt.nii.gz").get_data()
        pix_dim = img.header.structarr['pixdim'][1]
        img = np.array(img.get_data())
        seg = np.array(seg)

        pre_shape = img.shape[:-1]
        ratio = float(pix_dim/1.25)
        scale_vector = [ratio, ratio, 1]

        img = transform.rescale(img,
                                scale_vector,
                                order=1,
                                preserve_range=True,
                                multichannel=False,
                                mode='constant')
        seg = transform.rescale(seg,
                                scale_vector,
                                order=0,
                                preserve_range=True,
                                multichannel=False,
                                mode='constant')

        if self.augmentations is not None:
            img = img.transpose(2, 0, 1)
            seg = seg.transpose(2, 0, 1)
            img_c = np.zeros((img.shape[0], self.target_size[0], self.target_size[1]))
            seg_c = np.zeros((seg.shape[0], self.target_size[0], self.target_size[1]))

            for z in range(img.shape[0]):
                if img[z].min() > 0:
                    img[z] -= img[z].min()

                img_tmp, seg_tmp = self.augmentations(img[z].astype(np.uint32), seg[z].astype(np.uint8))
                img_tmp = augment_gamma(img_tmp)

                mu = img_tmp.mean()
                sigma = img_tmp.std()
                img_tmp = (img_tmp - mu) / (sigma+1e-10)
                img_c[z] = img_tmp
                seg_c[z] = seg_tmp

            img = img_c.transpose(1,2,0)
            seg = seg_c.transpose(1,2,0)

        img = torch.from_numpy(img).float()
        seg = torch.from_numpy(seg).long()

        data_dict = {
            "name": filename,
            "image": img,
            "mask": seg,
        }

        return data_dict

    def _transform(self, img, mask):
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)
            img = np.concatenate((img, img, img), axis=0)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        return img, mask


if __name__ == "__main__":
    from pprint import pprint

    train_augs = Compose([PaddingCenterCrop(256), RandomHorizontallyFlip(), RandomVerticallyFlip(), RandomRotate(180)])


    # root = "/home/hao/Downloads/COCO2CULane"
    #
    #
    # dataset_train = SideWalkData(
    #     root=root,
    #     split='train',
    #     k_split=1,
    #     augmentations=train_augs
    # )
    #
    # img = dataset_train[0]
    # # print(img)
    #
    # ac17_train = load2D(dataset_train, split='train', deform=True)
    # d = ac17_train.data
    # pprint(d)
    #
    # loader_train = data.DataLoader(
    #     ac17_train,
    #     batch_size=3,
    #     shuffle=True,
    #     num_workers=int(1),
    #     drop_last=True,
    #     pin_memory=True)
    #
    # pprint(loader_train)

    root = "/home/hao/Downloads/COCO2CULane"

    dataset_train = AC17Data(
        root=root,
        split='train',
        k_split=1,
        augmentations=train_augs
    )

    img = dataset_train[0]
    # print(img)

    ac17_train = load2D(dataset_train, split='train', deform=True)
    d = ac17_train.data
    x = d[0]['mask']
    seg, edge = x
    print("Seg")
    pprint(seg)

    print("Edge")
    pprint(edge)


    loader_train = data.DataLoader(
        ac17_train,
        batch_size=3,
        shuffle=True,
        num_workers=int(1),
        drop_last=True,
        pin_memory=True)

    # pprint(loader_train['mask'])




