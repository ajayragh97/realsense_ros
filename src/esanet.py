#!/home/ajay/anaconda3/envs/rgbd_segmentation/bin/python3.7

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import argparse
import sys
import numpy as np
# from multiprocessing import Pipe

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src.prepare_data import prepare_data



def get_gradients(img_rgb, img_depth, h, w):
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Inference)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--ckpt_path', type=str,
                        default="/home/ajay/cvbridge_build_ws/src/open3d_ros/src/trained_models/sunrgbd_r34_NBt1D_scenenet/sunrgbd/r34_NBt1D_scenenet.pth",
                        help='Path to the checkpoint of the trained model.')
    parser.add_argument('--depth_scale', type=float,
                        default=0.001,
                        help='Additional depth scaling factor to apply.')
    args = parser.parse_args()
    args.pretrained_on_imagenet = False
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

    # inference
  
    # load sample
    # img_rgb = _load_img(fp_rgb)
    # img_depth = _load_img(fp_depth).astype('float32') * args.depth_scale
    # h, w, _ = img_rgb.shape
    
    
    # preprocess sample

    sample = preprocessor({'image': img_rgb, 'depth': img_depth})

    # add batch axis and copy to device
    image = sample['image'][None].to(device)
    depth = sample['depth'][None].to(device)

    # apply network
    pred = model(image, depth)
    pred = F.interpolate(pred, (h, w),
                            mode='bilinear', align_corners=False)
    pred = pred.softmax(dim=1)
    # pred1 = torch.argmax(pred, dim=1)
    # pred1 = pred1.cpu().numpy().squeeze().astype(np.uint8)
    # pred_colored = dataset.color_label(pred1, with_void=False)
    # plt.imsave("/home/ajay/seg.png", pred_colored)
    pred = pred.cpu().detach().numpy().squeeze()

    return pred

# if __name__ == "__main__":
#     # Parse the RGB and depth image arguments from the command line
#     rgb_image_str = sys.stdin.readline().strip()
#     depth_image_str = sys.stdin.readline().strip()

#     # Convert the hex strings back to numpy arrays
#     rgb_image = np.frombuffer(bytes.fromhex(rgb_image_str), dtype=np.uint8).reshape((1280, 720, 3))
#     depth_image = np.frombuffer(bytes.fromhex(depth_image_str), dtype=np.uint16).reshape((1280, 720))

#     # Call the get_gradients function with the provided images
#     gradients = get_gradients(rgb_image, depth_image)
#     print(gradients.shape)

