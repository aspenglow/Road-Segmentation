# To make augmentation for training images and their groundtruth.

from imgaug import augmenters as iaa
import numpy as np
import imgaug as ia
import cv2
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def data_aug(img, gt):

    # get segmentation map object and corresponding image
    segmap = SegmentationMapsOnImage(gt, shape=gt.shape)
    # define transformation rule
    seq = iaa.Sequential([
        iaa.Fliplr(p=0.5),
        iaa.Flipud(p=0.5),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.ContrastNormalization((0.75, 1.5), per_channel=True),
        # iaa.Affine(
        #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #     rotate=(-25, 25),
        #     shear=(-8, 8))
        # iaa.Sharpen()
    ], random_order=True)

    # augmentation
    aug_img, aug_gt = seq(image=img, segmentation_maps=segmap)
    aug_gt = 255*aug_gt.get_arr()
    # save augmented images
    # img_name = "./data/training/augmented_images/images/" + str(i + 1) + ".png"
    # gt_name = "./data/training/augmented_images/groundtruth/" + str(i + 1) + ".png"
    # cv2.imwrite(img_name, aug_img)
    # cv2.imwrite(gt_name, 255*aug_gt.get_arr())

    return aug_img, aug_gt




