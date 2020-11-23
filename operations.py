import matplotlib.image as mpimg
import numpy as np
import os
from mask_to_submission import patch_to_label
from sklearn.metrics import f1_score
from PIL import Image


def compute_F1(pred, gt, args):
    # extract label list
    patch_pred = [img_crop(pred[i].detach().numpy(), args) for i in range(args.batch_size)]
    patch_gt = [img_crop(gt[i].detach().numpy(), args) for i in range(args.batch_size)]
    f1 = f1_score(np.array(patch_gt).ravel(), np.array(patch_pred).ravel())
    return f1


def load_image(infilename):
    # load image and rescale its entries to 0-255
    data = 255*mpimg.imread(infilename)
    return data


def load_images(img_dir):
    # load images in a directory
    files = os.listdir(img_dir)
    n = len(files)
    print("Loading " + str() + " images......")
    imgs = [load_image(img_dir + files[i]) for i in range(n)]

    return imgs


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def concatenate_images(img, gt_img):
    # Concatenate an image and its groundtruth
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, args, w=16, h=16):
    # extract patches from a given image
    list_labels = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            label = patch_to_label(im_patch, args)
            list_labels.append(label)
    return list_labels


def print_network(model):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))
