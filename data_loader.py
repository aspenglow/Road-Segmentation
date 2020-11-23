# TODO: validation problem
from torch.utils import data
from img_augment import *
from operations import *

class ImageLoader(data.Dataset):
    def __init__(self, args, mode):

        # initialize parameters
        self.mode = mode
        self.train_path = args.train_path
        self.gt_path = args.gt_path
        self.valid_path = args.valid_path
        self.test_path = args.test_path
        self.tr_files = os.listdir(args.train_path)
        self.gt_files = os.listdir(args.gt_path)
        self.te_files = os.listdir(args.test_path)

        # print dataset information
        print('build ' + mode + ' dataloader..............')
        if mode == 'train':
            print('found ' + str(len(self.tr_files)) + ' images......')
        elif mode == 'test':
            print('found ' + str(len(self.te_files)) + ' images......')
        elif mode == 'valid':
            print('found ' + str(len(self.tr_files)) + ' images......')

    def __getitem__(self, index):
        # load image and do preprocessing and augmentation
        if self.mode == 'train':
            img = load_image(self.train_path + self.tr_files[index])
            gt = load_image(self.gt_path + self.gt_files[index])
            # data augmentation
            # TODO: find a solution for augmentation
            # img, gt = data_aug(img, gt)
            gt[gt > 0] = 1
            return img, gt.astype(np.int32)

        elif self.mode == 'valid':
            # if validation then no augmentation
            img = load_image(self.train_path + self.tr_files[index])
            gt = load_image(self.gt_path + self.gt_files[index])
            gt[gt > 0] = 1

            return img, gt.astype(np.int32)


        elif self.mode == 'test':
            te_files = os.listdir(self.test_path + 'test_' + str(index + 1))
            img = load_image(self.test_path + 'test_' + str(index + 1) + te_files[0])

            return img

    def __len__(self):
        if self.mode == 'train':
            return len(self.tr_files)
        elif self.mode == 'valid':
            return len(self.tr_files)
        elif self.mode == 'test':
            return len(self.te_files)

