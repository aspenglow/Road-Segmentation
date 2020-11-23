import argparse
import pickle
import torch.optim as optim
from torch.utils import data
from U_Net import *
from operations import *
from mask_to_submission import *
from data_loader import ImageLoader



def run(args):
    # build dataset
    train_set = ImageLoader(args, mode='train')
    valid_set = ImageLoader(args, mode='valid')
    test_set = ImageLoader(args, mode='test')

    # build data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_set,
        batch_size=args.batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True)

    # build model and train
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = U_Net().to(device)
    optimizer = optim.Adam(model.parameters(), betas=(args.beta1, args.beta2))
    criterion = nn.BCELoss()
    losses = []
    F1 = []
    for epoch in range(args.training_epochs):
        avg_cost, avg_f1 = 0, 0
        total_batch = len(train_loader)
        for batch_idx, (img, gt) in enumerate(train_loader):
            t_img = img.permute(0, 3, 1, 2).to(device)
            gt = gt.to(device)
            optimizer.zero_grad()
            seg_out = model(t_img)
            seg_probs = torch.sigmoid(seg_out).squeeze(1)  # squeeze abundant dim
            # compute loss
            loss = criterion(seg_probs, gt.float())
            print('current loss:', loss)
            avg_cost += loss.data.numpy()/total_batch
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            # compute F1-score
            avg_f1 += compute_F1(seg_probs, gt, args)/total_batch

        losses.append(avg_cost)
        F1.append(avg_f1)

    # save loss and F1 score
    if args.output is not None:
        pickle.dump({"train_loss": losses, "F1_score": F1},
                    open(args.output, "wb"))

    # test model on validate set
    # TODO: validation
    # predict on test set
    image_filenames = []
    for batch_idx, img in enumerate(test_loader):
        pred = model(img)
        pred_probs = torch.sigmoid(pred)
        for i in range(args.batch_size):
            image_filenames.append(pred_probs[i])
    result_to_submission(args.result_path, image_filenames, args)  # the second param should be a list of image mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # variable args
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--training_epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of batch sizes')
    parser.add_argument('--beta1', type=float, default=0.9, help='first order decaying parameter')
    parser.add_argument('--beta2', type=float, default=0.999, help='second order decaying parameter')
    parser.add_argument('--aug_prob', type=float, default=0.3, help='augmentation probability')
    parser.add_argument('--foreground_threshold', type=float, default=0.25,
                        help='percentage of pixels > 1 required to assign a foreground label to a patch')
    # constant args
    parser.add_argument('--train_path', type=str, default='./data/training/images/')
    parser.add_argument('--gt_path', type=str, default='./data/training/groundtruth/')
    parser.add_argument('--valid_path', type=str, default='./data/valid/')  # TODO: validation set
    parser.add_argument('--test_path', type=str, default='./data/test_set_images/')
    parser.add_argument('--result_path', type=str, default='./output/my_submission.csv')
    parser.add_argument('--output', type=str, default="./output/result.pkl", help='Output file to save training loss\
       and accuracy.')

    args = parser.parse_args()
    run(args)
