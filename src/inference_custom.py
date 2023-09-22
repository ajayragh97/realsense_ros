# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
from glob import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src.prepare_data import prepare_data


def _load_img(fp):
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
def _load_depth(fp):
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    normalized_image = (((img - img.min()) / (img.max() - img.min())) * 255).astype('float32')
    return normalized_image


if __name__ == '__main__':
    # arguments
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Inference)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--ckpt_path', type=str,
                        required=True,
                        help='Path to the checkpoint of the trained model.')
    parser.add_argument('--depth_scale', type=float,
                        default=1.0,
                        help='Additional depth scaling factor to apply.')
    args = parser.parse_args()

    # dataset
    args.pretrained_on_imagenet = False  # we are loading other weights anyway
    dataset, preprocessor = prepare_data(args, with_input_orig=True)
    n_classes = dataset.n_classes_without_void

    # model and checkpoint loading
    model, device = build_model(args, n_classes=n_classes)
    checkpoint = torch.load(args.ckpt_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded checkpoint from {}'.format(args.ckpt_path))

    model.eval()
    model.to(device)

    # get samples
    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'samples')
    rgb_filepaths = sorted(glob("/home/ajay/work/msc_project/git/MSR_Project/Results/RGBD/rgb/*.jpg"))
    depth_filepaths = sorted(glob("/home/ajay/work/msc_project/git/MSR_Project/Results/RGBD/depth/*.png"))
    assert args.modality == 'rgbd', "Only RGBD inference supported so far"
    assert len(rgb_filepaths) == len(depth_filepaths)
    filepaths = zip(rgb_filepaths, depth_filepaths)

    # inference
    for fp_rgb, fp_depth in filepaths:
        # load sample
        name = os.path.basename(fp_rgb)
        print(name)
        img_rgb = _load_img(fp_rgb)
        img_depth = _load_depth(fp_depth).astype('float32') * args.depth_scale
        h, w, _ = img_rgb.shape
        # print(h, w)
        # preprocess sample
        sample = preprocessor({'image': img_rgb, 'depth': img_depth})

        # add batch axis and copy to device
        image = sample['image'][None].to(device)
        depth = sample['depth'][None].to(device)

        # apply network
        pred = model(image, depth)
        # print(pred.shape)
        pred = F.interpolate(pred, (h, w),
                             mode='bilinear', align_corners=False)
        

        
        
        # 
            # print(i)
        # print(pred_gradients)   
        # pred = torch.argmax(pred, dim=1)
        pred = pred.softmax(dim=1)
        # pred = pred.cpu().numpy().squeeze().astype(np.uint8)
        pred = pred.cpu().detach().numpy().squeeze()
        

        # # show result
        # pred_colored = dataset.color_label(pred, with_void=False)
        # plt.imsave("/home/ajay/work/msc_project/ms3/test/seg/"+name, pred_colored)

        # fig, axs = plt.subplots(1, 3, figsize=(16, 3))
        # [ax.set_axis_off() for ax in axs.ravel()]
        # axs[0].imshow(img_rgb)
        # axs[1].imshow(img_depth, cmap='gray')
        # axs[2].imshow(pred_colored)

        # plt.suptitle(f"Image: ({os.path.basename(fp_rgb)}, "
        #              f"{os.path.basename(fp_depth)}), Model: {args.ckpt_path}")
        # # plt.savefig('./result.jpg', dpi=150)
        # plt.show()

        pred_gradients = np.zeros((h*w, pred.shape[0]))
        count = 0
        # print(h, w)
        for i in range(w):
            for j in range(h):
                pred_gradients[count, :] = pred[:, j, i]
                count +=1
        name = os.path.splitext(name)[0] + ".npy"

        # np.save(f"/home/ajay/work/msc_project/git/gradients/{name}", pred_gradients)

