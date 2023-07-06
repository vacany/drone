# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def get_model():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)


    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth')        
    # logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    # for k, _ in pretrained_dict.items():
        # logger.info(
            # '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # gpus = list(config.GPUS)
    # model = nn.DataParallel(model, device_ids=gpus).cuda()
    return model


if __name__ == '__main__':

    color_dict = {
     'ScrubBrush': (222, 148, 80),
     'CoffeeMachine': (147, 71, 238),
     'RemoteControl': (187, 19, 208),
     'TissueBox': (98, 43, 249),
     'DishSponge': (166, 58, 136),
     'SideTable': (202, 45, 114),
     'CounterTop': (103, 209, 30),
     'Spoon': (235, 57, 90),
     'CoffeeTable': (18, 14, 75),
     'Bed': (209, 156, 101),
     'VacuumCleaner': (230, 13, 166),
     'Window': (200, 150, 134),
     'Watch': (242, 6, 88),
     'GarbageBag': (250, 186, 207),
     'PaperTowelRoll': (144, 173, 28),
     'TowelHolder': (232, 28, 225),
     'Television': (27, 245, 217),
     'Sofa': (82, 143, 39),
     'SoapBottle': (168, 222, 137),
     'Boots': (121, 126, 101),
     'SaltShaker': (36, 222, 26),
     'ArmChair': (96, 52, 68),
     'Desk': (14, 120, 179),
     'Pot': (132, 237, 87),
     'Tomato': (119, 189, 121),
     'LightSwitch': (11, 51, 121),
     'Curtains': (6, 62, 102),
     'DeskLamp': (99, 164, 25),
     'Lettuce': (203, 156, 88),
     'Pan': (246, 212, 161),
     'RoomDecor': (216, 96, 246),
     'CellPhone': (227, 98, 136),
     'Floor': (243, 246, 208),
     'BaseballBat': (171, 20, 38),
     'Dumbbell': (45, 57, 144),
     'Cup': (35, 71, 130),
     'ToiletPaper': (162, 204, 152),
     'SinkBasin': (80, 192, 81),
     'ShowerGlass': (80, 68, 237),
     'Fork': (54, 200, 25),
     'SprayBottle': (89, 126, 121),
     'TeddyBear': (229, 73, 134),
     'StoveBurner': (156, 249, 101),
     'FloorLamp': (253, 73, 35),
     'Stool': (13, 54, 156),
     'Potato': (187, 142, 9),
     'Toaster': (55, 33, 114),
     'Ottoman': (160, 135, 174),
     'HandTowel': (182, 187, 236),
     'Bottle': (64, 80, 115),
     'HandTowelHolder': (58, 218, 247),
     'Laptop': (20, 107, 222),
     'Kettle': (7, 83, 48),
     'Pillow': (217, 193, 130),
     'Candle': (233, 102, 178),
     'ShelvingUnit': (125, 226, 119),
     'Shelf': (39, 54, 158),
     'DogBed': (106, 193, 45),
     'Ladle': (174, 98, 216),
     'Faucet': (21, 38, 98),
     'ButterKnife': (135, 147, 55),
     'Knife': (211, 157, 122),
     'Dresser': (51, 128, 146),
     'AluminumFoil': (181, 163, 89),
     'Poster': (145, 87, 153),
     'Pen': (239, 130, 152),
     'TennisRacket': (138, 71, 107),
     'Towel': (170, 186, 210),
     'Newspaper': (19, 196, 2),
     'ShowerHead': (248, 167, 29),
     'Bowl': (209, 182, 193),
     'Pencil': (177, 226, 23),
     'Blinds': (214, 223, 197),
     'Footstool': (74, 187, 51),
     'ShowerCurtain': (60, 12, 39),
     'TVStand': (94, 234, 136),
     'Plate': (188, 154, 128),
     'Drawer': (155, 30, 210),
     'GarbageCan': (225, 40, 55),
     'Mirror': (36, 3, 222),
     'Book': (43, 31, 148),
     'BathtubBasin': (109, 206, 121),
     'Plunger': (74, 209, 56),
     'AlarmClock': (184, 20, 170),
     'Vase': (83, 152, 69),
     'SoapBar': (43, 97, 155),
     'Chair': (166, 13, 176),
     'BasketBall': (97, 58, 36),
     'LaundryHamper': (35, 109, 26),
     'Toilet': (21, 27, 163),
     'Sink': (30, 181, 88),
     'KeyChain': (27, 54, 18),
     'Microwave': (54, 96, 202),
     'Bathtub': (59, 170, 176),
     'ToiletPaperHanger': (124, 32, 10),
     'Box': (60, 252, 230),
     'Egg': (240, 75, 163),
     'Cloth': (110, 184, 56),
     'TableTopDecor': (126, 204, 158),
     'Cabinet': (210, 149, 89),
     'DiningTable': (83, 33, 33),
     'Fridge': (91, 156, 207),
     'Apple': (159, 98, 144),
     'WateringCan': (147, 67, 249),
     'CD': (65, 112, 172),
     'Mug': (8, 94, 186),
     'StoveKnob': (106, 252, 95),
     'CreditCard': (56, 235, 12),
     'Desktop': (35, 16, 64),
     'Safe': (198, 238, 160),
     'PepperShaker': (5, 204, 214),
     'Spatula': (30, 98, 242),
     'ShowerDoor': (36, 253, 61),
     'HousePlant': (73, 144, 213),
     'WineBottle': (53, 130, 252),
     'Statue': (243, 75, 41),
     'Bread': (18, 150, 252),
     'Painting': (40, 117, 236)}

    color_array = np.array((list(color_dict.values())))
    model = get_model()

    # running script
    # python tools/test.py --cfg experiments/pascal_ctx/seg_hrnet_w48_cls59_520x520_sgd_lr1e-3_wd1e-4_bs_16_epoch200.yaml

    # https://www.cs.stanford.edu/~roozbeh/pascal-context/59_labels.txt
    label_map_lines = open('label_map_59.txt').readlines()
    label_map = {int(x.split(":")[0]) : x.split(" ")[-1].strip() for x in label_map_lines}

    # Next tasks:

    # run preprocessing on semantickitti etc.
    # storing the labels and confidences to voxel map based on point cloud coordinates, or map
    # video of constructing accumulated confidence map
    # run prior generation on the delft dataset
    # motion segmentation and motion flow on the points in visual for thursday - nope
    # That should be enough for presentation

    import glob
    from PIL import Image
    import matplotlib.pyplot as plt
    image_files = sorted(glob.glob(os.path.expanduser("~") + '/data/drone/showcase/rgb/*.png'))

    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]

    model = model.cuda()

    # from datasets.delft.drone import Delft_Sequence
    # from datasets.structures import realsense, rgbd

    # cx, cy, fx, fy = realsense.cx, realsense.cy, realsense.fx, realsense.fy

    # sequence = Delft_Sequence(0)

    folders = ['hrnet_pts', 'hrnet_confidence', 'hrnet_confidence_npy', 'hrnet_seg']
    for folder in folders:
        os.makedirs('/home/vacekpa2/data/drone/showcase/' + folder, exist_ok=True)

    from tqdm import tqdm

    rgb_files = sorted(glob.glob('/home/vacekpa2/data/drone/showcase/rgb/*'))
    depth_files = sorted(glob.glob('/home/vacekpa2/data/drone/showcase/depth/*'))

    # try HRNET, if not, then retrain it on SUNRGBD?
    # framework for annotating data?
    # just something running on the drone. The rest is running from the simulator anyway ...

    for idx in tqdm(range(len(rgb_files))):
        # Preload data

        rgb = Image.open(rgb_files[idx])
        depth = Image.open(depth_files[idx])


        # For inference
        image = Image.fromarray(rgb)
        image = image.resize((520, 520))
        image = np.asarray(image)

        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= mean
        image /= std

        x = torch.tensor(image, dtype=torch.float).permute(2,0,1).unsqueeze(0).cuda()
        out = model(x)


        # store confidence
        confidence = torch.max(torch.softmax(out, dim=1), dim=1)[0].cpu().detach().numpy()[0]
        confidence_file = sequence.sequence_path + '/hrnet_confidence/' + str(idx).zfill(6) + '.png'

        confidence_file_npy = sequence.sequence_path + '/hrnet_confidence_npy/' + str(idx).zfill(6) + '.npy'
        np.save(confidence_file[:-4] + '.npy', confidence)

        plt.imshow(confidence, cmap='jet')
        plt.savefig(confidence_file)
        plt.close()

        # store seg label
        seg_label = torch.argmax(out, dim=1)
        seg_img = color_array[seg_label[0].detach().cpu().numpy()]
        seg_final_image = Image.fromarray(seg_img.astype('uint8'))
        out_file = sequence.sequence_path + '/hrnet_seg/' + str(idx).zfill(6) + '.png'
        seg_final_image.save(out_file)

        # store point cloud
        # resize depth and rgb
        rgb_img = Image.fromarray(rgb)
        rgb_img = rgb_img.resize((520, 520), resample=Image.NEAREST)
        rgb = np.asarray(rgb_img) / 255.0

        # Depth
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.resize((520, 520), resample=Image.NEAREST)
        depth = np.asarray(depth_img)
        depth = depth / 1024.0

        # seg_img
        seg_img = Image.fromarray(seg_img.astype('uint8'))
        seg_img = seg_img.resize((520, 520), resample=Image.NEAREST)
        seg_img = np.asarray(seg_img) / 255.0

        # confidence
        confidence_img = Image.fromarray(confidence)
        confidence_img = confidence_img.resize((520, 520), resample=Image.NEAREST)
        confidence = np.asarray(confidence_img)

        # seg_label
        seg_label = seg_label.cpu().detach().numpy()[0]
        seg_label = Image.fromarray(seg_label.astype('uint8'))
        seg_label = seg_label.resize((520, 520), resample=Image.NEAREST)
        seg_label = np.asarray(seg_label)


        # project point cloud
        pts = rgbd.project_depth_to_pcl(depth, rgb, seg_img, confidence, seg_label, cx, cy, fx, fy)    # later seg_img for seg_label

        np.save(sequence.sequence_path + '/hrnet_pts/' + str(idx).zfill(6), pts)
