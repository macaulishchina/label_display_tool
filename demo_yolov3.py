# coding=utf-8
import sys
import os
import numpy as np
import random
import math

import cv2
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.parallel import DataParallel
import cPickle as pickle
import time
import argparse
from PIL import Image, ImageFont, ImageDraw

from baseline.model.DeepMAR import DeepMAR_ResNet50, DeepMAR_ResNet152
from baseline.utils.utils import str2bool
from baseline.utils.utils import save_ckpt, load_ckpt
from baseline.utils.utils import load_state_dict
from baseline.utils.utils import set_devices
from baseline.utils.utils import set_seed


class Config(object):
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
        parser.add_argument('--set_seed', type=str2bool, default=False)
        # model
        parser.add_argument('--resize', type=eval, default=(224, 224))
        parser.add_argument('--last_conv_stride', type=int, default=2, choices=[1, 2])
        # demo image
        parser.add_argument('--demo_image', type=str, default='./dataset/demo/demo_image.png')
        ## dataset parameter
        parser.add_argument('--dataset', type=str, default='pa100k',
                            choices=['peta', 'rap', 'pa100k'])
        # utils
        parser.add_argument('--load_model_weight', type=str2bool, default=True)
        parser.add_argument('--model_weight_file', type=str,
                            default='../../exp/deepmar_resnet152/pa100k/partition0/run1/model/ckpt_epoch50.pth')
        args = parser.parse_args()

        # gpu ids
        self.sys_device_ids = args.sys_device_ids

        # random
        self.set_seed = args.set_seed
        if self.set_seed:
            self.rand_seed = 0
        else:
            self.rand_seed = None
        self.resize = args.resize
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # utils
        self.load_model_weight = args.load_model_weight
        self.model_weight_file = args.model_weight_file
        if self.load_model_weight:
            if self.model_weight_file == '':
                print 'Please input the model_weight_file if you want to load model weight'
                raise ValueError
        # dataset
        datasets = dict()
        datasets['peta'] = '../../dataset/peta/peta_dataset.pkl'
        datasets['rap'] = './dataset/rap/rap_dataset.pkl'
        datasets['pa100k'] = '../../dataset/pa100k/pa100k_dataset.pkl'

        if args.dataset in datasets:
            dataset = pickle.load(open(datasets[args.dataset]))
        else:
            print '%s does not exist.' % (args.dataset)
            raise ValueError
        self.att_list = [dataset['att_name'][i] for i in dataset['selected_attribute']]

        # demo image
        self.demo_image = args.demo_image

        # model
        model_kwargs = dict()
        model_kwargs['num_att'] = len(self.att_list)
        model_kwargs['last_conv_stride'] = args.last_conv_stride
        self.model_kwargs = model_kwargs


# 创建黑色画布
def create_canvas(height, width, channel):
    # create a black use numpy,size is:512*512
    img = np.zeros((height, width, channel), np.uint8)
    # 用嘿色填充
    img.fill(0)
    return img


def save_eval_result(txtPath, info):
    f = open(txtPath, 'a')
    f.write(info)
    f.close()


def filter_txt_by_frame(det_path, filter_txt_by_frame_path):
    uniq_frame = list(range(1, 1051))
    for frame in uniq_frame:
        with open(det_path) as f:
            for line in f:
                frame_txt = line.split(",")[0]
                bb_left = int(line.split(",")[2])
                bb_top = int(line.split(",")[3])
                bb_width = int(line.split(",")[4])
                bb_height = int(line.split(",")[5])

                MOT_type = line.split(",")[7]

                xmin, ymin, xmax, ymax = bb_left, bb_top, bb_left + bb_width, bb_top + bb_height
                line = frame_txt + "," + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(
                    ymax) + "," + MOT_type

                if str(frame_txt) == str(frame):
                    txt_name = os.path.join(filter_txt_by_frame_path, (str(frame).zfill(6) + ".txt"))
                    save_eval_result(txt_name, line + "\n")


if __name__ == '__main__':
    ### main function ###
    cfg = Config()

    # dump the configuration to log.
    import pprint

    print('-' * 60)
    print('cfg.__dict__')
    pprint.pprint(cfg.__dict__)
    print('-' * 60)

    # set the random seed
    if cfg.set_seed:
        set_seed(cfg.rand_seed)
    # init the gpu ids
    set_devices(cfg.sys_device_ids)

    # dataset
    normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
    test_transform = transforms.Compose([
        transforms.Resize(cfg.resize),
        transforms.ToTensor(),
        normalize, ])

    ### Att model ###
    model = DeepMAR_ResNet152(**cfg.model_kwargs)

    # load model weight if necessary
    if cfg.load_model_weight:
        map_location = (lambda storage, loc: storage)
        ckpt = torch.load(cfg.model_weight_file, map_location=map_location)
        model.load_state_dict(ckpt['state_dicts'][0])

    model.cuda()
    model.eval()

    # load one image
    # load one image
    # 从视频中加载每一帧
    import torchvision.transforms.functional as F

    # det_path = "/disk3/gantao.xiao/MOT/MOT_data/train/MOT16-04/gt/gt.txt"
    # filter_txt_by_frame_path = "/disk3/gantao.xiao/MOT/MOT_data/train/MOT16-04/gt/filter_txt_by_frame"
    # img_base_path = "/disk2/gantao.xiao/other_task/demo_input/demo5/img1"
    img_base_path = "/disk2/gantao.xiao/other_task/demo_input/demo7/img1"
    txt_base_path = "/disk2/gantao.xiao/other_task/demo_input/demo7/numpy_mat"
    img_path = os.listdir(img_base_path)
    img_path.sort(key=lambda x: int(x[:-4]))
    # 透明度
    alpha = 0.6

    # 根据frame 筛选gt.txt
    # filter_txt_by_frame(det_path, filter_txt_by_frame_path)

    for txt_name in os.listdir(txt_base_path):
        # 一帧一帧的处理
        # print txt_name
        jpg_name = str(int(txt_name.split(".")[0]) + 1).zfill(6) + ".jpg"
        print jpg_name
        # print txt_name
        jpg_path = os.path.join(img_base_path, jpg_name)
        frame = cv2.imread(jpg_path)

        with open(os.path.join(txt_base_path, txt_name)) as f:
            for line in f:
                xmin = int(line.split(" ")[0])
                ymin = int(line.split(" ")[1])
                xmax = int(line.split(" ")[2])
                ymax = int(line.split(" ")[3])
                type_MOT = line.split(" ")[4].split("\n")[0]

                width = (xmax - xmin)
                height = (ymax - ymin)

                # 往左上移动
                xmin = xmin - width / 2
                ymin = ymin - height / 2
                xmax = xmax - width / 2
                ymax = ymax - height / 2

                # 如果是行人，传入deep mar模型跑出类别
                if str(type_MOT) == "0":
                    # print "yes"
                    # frame是numpy.array，转化为PIL.Image
                    frame_img = Image.fromarray(frame.astype('uint8')).convert('RGB')
                    single_region = F.crop(frame_img, xmin, ymin, xmax - xmin, ymax - ymin)
                    # img = Image.open(cfg.demo_image)
                    img_trans = test_transform(single_region)
                    img_trans = torch.unsqueeze(img_trans, dim=0)
                    img_var = Variable(img_trans).cuda()
                    # 将single_region传递给model，得到score
                    score = model(img_var).data.cpu().numpy()

                    # show the score in command line
                    # for idx in range(len(cfg.att_list)):
                    #     if score[0, idx] >= 0:
                    #         print '%s: %.2f' % (cfg.att_list[idx], score[0, idx])
                    max_len = 0
                    flag = 0
                    for idx in range(len(cfg.att_list)):
                        if score[0, idx] >= 2:
                            flag = 1
                            if len(str(cfg.att_list[idx])) > max_len:
                                max_len = len(str(cfg.att_list[idx]))

                    if flag == 0:
                        max_len = 3

                    # 红色检测框
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    x = xmax + 10
                    y = ymin
                    # 白色类别
                    for idx in range(len(cfg.att_list)):
                        # 判断性别
                        if cfg.att_list[idx] == "Female" and score[0, idx] < 1:
                            overlay = frame.copy()
                            output = frame.copy()
                            cv2.rectangle(overlay, (xmax + 8, y - 16),
                                          (xmax + (max_len + 4) * 5, y - 3),
                                          (0, 0, 0), -1)
                            frame = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, frame)
                            cv2.putText(frame, "Male", (x, y - 6), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (255, 255, 255),
                                        1,cv2.LINE_AA)
                            y += 15

                        elif score[0, idx] >= 2:
                            if cfg.att_list[idx] == "LongCoat" or cfg.att_list[idx] == "Skirt&Dress":
                                pass
                            else:

                                txt = '%s' % (cfg.att_list[idx])
                                # 先画黑色透明背景
                                overlay = frame.copy()
                                output = frame.copy()
                                cv2.rectangle(overlay, (xmax + 8, y - 16),
                                              (xmax + (max_len + 4) * 5, y - 3),
                                              (0, 0, 0), -1)
                                frame = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, frame)
                                # 再画白色字
                                cv2.putText(frame, txt, (x, y - 6), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (255, 255, 255),
                                            1,cv2.LINE_AA)
                                y += 15
                # 如果是其他类别
                else:
                    # pass
                    # 黄色检测框
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
            result_path = "/disk3/gantao.xiao/PAR/pedestrian-attribute-recognition-pytorch-master/script/experiment/demo7/"
            result_img_name = result_path + jpg_name
            # result_img_name = "./" + jpg_name
            cv2.imwrite(result_img_name, frame)

        # # 只处理第一帧
        # break
