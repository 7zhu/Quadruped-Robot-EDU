import argparse
import csv
import os
import platform
import sys
import shutil
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = os.path.join(ROOT, "yolov5")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import pyrealsense2 as rs
import numpy as np
import cv2
import random
import torch
import torch.backends.cudnn as cudnn # cuda模块
import time

from yolov5.utils.torch_utils import select_device
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_boxes,
    xyxy2xywh, strip_optimizer, set_logging, Profile, LOGGER
    )
from yolov5.utils.augmentations import letterbox
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from yolov5.models.experimental import attempt_load

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
#==========================================================================================#


#==========================================================================================#
# plot_one_box()
# 输入  (参数位置x 在哪张画面 颜色 标签 线条粗细)
# 输出  (在画面上画上框框)
#==========================================================================================#
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # 字体大小
    color = color or [random.randint(0, 255) for _ in range(3)]	#随机颜色
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)   #通过确定对角线位置来画矩形框
    if label:
        tf = max(tl - 1, 1)  # font thickness字体粗细
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                    #(图片img,“文本内容”,(左下角坐标),字体,字体大小,(颜色)，线条粗细，线条类型)




#==========================================================================================#
# detect()
# 检测的主题函数
#==========================================================================================#
def detect(opt, save_img=False):
    # 加载参数
    out, source, weights, view_img, save_txt, imgsz, half = \
        opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.half
    webcam = source == '0' or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    set_logging()	#生成日志
    device = select_device(opt.device)
    if os.path.exists(out):  # output dir
        shutil.rmtree(out)  # delete dir
    os.makedirs(out)  # make new dir

    half = True if half else (device.type!='cpu') # True means using FLOAT16
    model = attempt_load(weights, device=device) # torch.load  DetectMultiBackend(main)
    imgsz = check_img_size(imgsz, s=model.stride.max()) # 验证图片大小
    if half:
        model.half()  # to FP16
    
    view_img = True
    cudnn.benchmark = True  # 加快常量图像大小推断
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # start depth camera
    ChannelFactoryInitialize(0)
    webclient = VideoClient()  # Create a video client
    webclient.SetTimeout(3.0)
    webclient.Init()
    webcode, webdata = webclient.GetImageSample()

    # warmup
    img_warmup = torch.zeros((1, 3, imgsz, imgsz), device=device)  
    _ = model(img_warmup.half() if half else img_warmup) if device.type != 'cpu' else None  # run once
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    while True:
        # get image
        # frames = pipeline.wait_for_frames() #等待最新的影像，wait_for_frames返回的是一個合成的影像
        # frames = align_to_color.process(frames) #将上图获取视频帧对齐
        # depth_frame = frames.get_depth_frame()
        # # depth_image = np.asanyarray(depth_frame.get_data())
        # color_frame = frames.get_color_frame()
        # color_image = np.asanyarray(color_frame.get_data()) # (480, 640, 3)
        # mask = np.zeros([color_image.shape[0], color_image.shape[1]], dtype=np.uint8)
        # mask[0:480, 320:640] = 255
        webcode, webdata = webclient.GetImageSample()
        image_data = np.frombuffer(bytes(webdata), dtype=np.uint8)
        imagecv = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        image_re = cv2.resize(imagecv, (640,480), interpolation=cv2.INTER_AREA)
        color_image = np.asanyarray(image_re) # (480, 640, 3)
        # print(color_image.shape)


        # img pre-process
        with dt[0]:
            sources = [source]
            imgs = [None]
            path = sources
            imgs[0] = color_image
            im0s = imgs.copy()
            img = [letterbox(x, new_shape=imgsz)[0] for x in im0s]  # img: 进行resize + pad之后的图片
            img = np.stack(img, 0)  #沿着0dim进行堆叠 此时shape已经是[batch_size, h, w, channel]
            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to 3x416x416, uint8 to float32
            img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)
                            # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            img = torch.from_numpy(img).to(device) #将numpy转为pytorch的tensor,并转移到运算设备上计算
            # 如果图片是3维(RGB) 就在前面添加一个维度1当中batch_size=1
            # 因为输入网络的图片需要是4为的 [batch_size, channel, w, h]
            if len(img.shape) == 3:
                img = img[None]

        # detect
        with dt[1]:
            pred = model(img,augment=opt.augment)[0]
        
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # results process
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            have_car = False

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    if c == 2:
                        have_car = True
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化为 xywh
                    line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"
                    # label = f"{names[int(cls)]} {confidence_str}:{np.mean(distance_list)}m"
                    label = '%s%s' % (names[int(cls)], confidence_str)
                    if have_car:
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
        
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # while out



#==========================================================================================#
# man!!!
#==========================================================================================#
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./yolov5/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--half', default=True, help='use float16')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad(): # 只做推理时不需要计算梯度，只用前向传播，可以省显存
        detect(opt)

if __name__ == '__main__':
    main()
    