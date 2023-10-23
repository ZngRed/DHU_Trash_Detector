import argparse
import os
import sys
from pathlib import Path
import time
import cv2
import numpy as np

import sys
from PyQt5 import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QApplication, QMainWindow
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import torch

import serial
# from queue import Queue
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from multiprocessing import shared_memory

inf = -1

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        MainWindow.resize(1920, 1080)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Geometrys
        title_x, title_y = 5, 5
        nums_x, nums_y, nums_w, nums_h = 20, 135, 430, 100
        infm_x, infm_y, infm_w, infm_h = 20, 400, 710, 600
        video_x, video_y = 768, 400
        warn_x, warn_y = 970, 70
        camera_x, camera_y, camera_w, camera_h = 1420, 20, 480, 360

        # num
        self.anum = 0
        self.bnum = 0
        self.cnum = 0
        self.dnum = 0
        # Title
        self.Title = QtWidgets.QLabel(self.centralwidget)
        self.Title.setGeometry(QtCore.QRect(title_x + 377, title_y - 5, 510, 120))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 106, 106))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 106, 106))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(190, 190, 190))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.Title.setPalette(palette)
        self.Title.setLocale(QtCore.QLocale(QtCore.QLocale.Chinese, QtCore.QLocale.China))
        self.Title.setTextFormat(QtCore.Qt.RichText)
        self.Title.setObjectName("Title")
        # aRecyclable
        self.aRecyclable = QtWidgets.QProgressBar(self.centralwidget)
        self.aRecyclable.setGeometry(QtCore.QRect(nums_x, nums_y, nums_w, nums_h))
        self.aRecyclable.setFormat("可回收垃圾: %v")
        self.aRecyclable.setStyleSheet("font-size:44pt;")
        self.aRecyclable.setProperty("value", self.anum)
        self.aRecyclable.setObjectName("aRecyclable")
        # bHarmful
        self.bHarmful = QtWidgets.QProgressBar(self.centralwidget)
        self.bHarmful.setGeometry(QtCore.QRect(nums_x + 460, nums_y, nums_w, nums_h))
        self.bHarmful.setFormat("有害垃圾: %v")
        self.bHarmful.setStyleSheet("font-size:44pt;")
        self.bHarmful.setProperty("value", self.bnum)
        self.bHarmful.setObjectName("bHarmful")
        # cKitchen
        self.cKitchen = QtWidgets.QProgressBar(self.centralwidget)
        self.cKitchen.setGeometry(QtCore.QRect(nums_x, nums_y + 130, nums_w, nums_h))
        self.cKitchen.setFormat("其他垃圾: %v")
        self.cKitchen.setStyleSheet("font-size:44pt;")
        self.cKitchen.setProperty("value", self.cnum)
        self.cKitchen.setObjectName("cKitchen")
        # dOther
        self.dOther = QtWidgets.QProgressBar(self.centralwidget)
        self.dOther.setGeometry(QtCore.QRect(nums_x + 460, nums_y + 130, nums_w, nums_h))
        self.dOther.setFormat("厨余垃圾: %v")
        self.dOther.setStyleSheet("font-size:44pt;")
        self.dOther.setProperty("value", self.dnum)
        self.dOther.setObjectName("dOther")

        # back_Warn
        self.back_Warn = QtWidgets.QTextBrowser(self.centralwidget)
        self.back_Warn.setGeometry(QtCore.QRect(warn_x - 10, warn_y - 5, 420, 300))
        self.back_Warn.setObjectName("back_Warn")
        # WarnRecyclable
        self.WarnRecyclable = QtWidgets.QLabel(self.centralwidget)
        self.WarnRecyclable.setGeometry(QtCore.QRect(warn_x, warn_y, 300, 70))
        self.WarnRecyclable.setTextFormat(QtCore.Qt.RichText)
        self.WarnRecyclable.setObjectName("WarnRecyclable")
        # WarnHarmful
        self.WarnHarmful = QtWidgets.QLabel(self.centralwidget)
        self.WarnHarmful.setGeometry(QtCore.QRect(warn_x, warn_y + 70, 250, 70))
        self.WarnHarmful.setTextFormat(QtCore.Qt.RichText)
        self.WarnHarmful.setObjectName("WarnHarmful")
        # WarnKitchen
        self.WarnKitchen = QtWidgets.QLabel(self.centralwidget)
        self.WarnKitchen.setGeometry(QtCore.QRect(warn_x, warn_y + 140, 250, 70))
        self.WarnKitchen.setTextFormat(QtCore.Qt.RichText)
        self.WarnKitchen.setObjectName("WarnKitchen")
        # WarnOther
        self.WarnOther = QtWidgets.QLabel(self.centralwidget)
        self.WarnOther.setGeometry(QtCore.QRect(warn_x, warn_y + 210, 250, 70))
        self.WarnOther.setTextFormat(QtCore.Qt.RichText)
        self.WarnOther.setObjectName("WarnOther")
        # ifWarnR
        self.ifWarnR = QtWidgets.QLabel(self.centralwidget)
        self.ifWarnR.setGeometry(QtCore.QRect(warn_x + 300, warn_y, 100, 70))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(164, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(164, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(190, 190, 190))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.ifWarnR.setPalette(palette)
        self.ifWarnR.setFrameShadow(QtWidgets.QFrame.Plain)
        self.ifWarnR.setTextFormat(QtCore.Qt.RichText)
        self.ifWarnR.setObjectName("ifWarnR")
        # ifWarnH
        self.ifWarnH = QtWidgets.QLabel(self.centralwidget)
        self.ifWarnH.setGeometry(QtCore.QRect(warn_x + 300, warn_y + 70, 100, 70))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(164, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(164, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(190, 190, 190))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.ifWarnH.setPalette(palette)
        self.ifWarnH.setFrameShadow(QtWidgets.QFrame.Plain)
        self.ifWarnH.setTextFormat(QtCore.Qt.RichText)
        self.ifWarnH.setObjectName("ifWarnH")
        # ifWarnK
        self.ifWarnK = QtWidgets.QLabel(self.centralwidget)
        self.ifWarnK.setGeometry(QtCore.QRect(warn_x + 300, warn_y + 140, 100, 70))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(164, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(164, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(190, 190, 190))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.ifWarnK.setPalette(palette)
        self.ifWarnK.setFrameShadow(QtWidgets.QFrame.Plain)
        self.ifWarnK.setTextFormat(QtCore.Qt.RichText)
        self.ifWarnK.setObjectName("ifWarnK")
        # ifWarnO
        self.ifWarnO = QtWidgets.QLabel(self.centralwidget)
        self.ifWarnO.setGeometry(QtCore.QRect(warn_x + 300, warn_y + 210, 100, 70))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(164, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(164, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(190, 190, 190))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.ifWarnO.setPalette(palette)
        self.ifWarnO.setFrameShadow(QtWidgets.QFrame.Plain)
        self.ifWarnO.setTextFormat(QtCore.Qt.RichText)
        self.ifWarnO.setObjectName("ifWarnO")

        # Information
        self.Information = QtWidgets.QTextBrowser(self.centralwidget)
        self.Information.setGeometry(QtCore.QRect(infm_x, infm_y, infm_w, infm_h))
        self.Information.setStyleSheet("font-size:20pt;")
        self.Information.setObjectName("Information")
        # Background
        self.Background = QtWidgets.QListView(self.centralwidget)
        self.Background.setGeometry(QtCore.QRect(0, 0, 1920, 1080))
        self.Background.setStyleSheet("background-image: url(\"/home/dhu/TrashDetector/icons/background.jpg\");")
        self.Background.setObjectName("Background")
        # LOGO
        self.LOGO = QtWidgets.QListView(self.centralwidget)
        self.LOGO.setGeometry(QtCore.QRect(title_x, title_y, 352, 108))
        self.LOGO.setStyleSheet("background-image: url(\"/home/dhu/TrashDetector/icons/LOGO.png\");")
        self.LOGO.setObjectName("LOGO")
        # videoWidget
        self.videoWidget = QVideoWidget(self.centralwidget)
        self.videoWidget.setGeometry(QtCore.QRect(video_x, video_y, 1152, 648))  # Adjust the geometry as needed
        self.videoWidget.setObjectName("videoWidget")
        # mediaPlayer
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.mediaStatusChanged.connect(self.restart)
        # Camera
        self.Camera = QtWidgets.QLabel(self.centralwidget)
        self.Camera.setGeometry(QtCore.QRect(camera_x, camera_y, camera_w, camera_h))  # Adjust the geometry as needed
        self.Camera.setObjectName("Camera")

        self.Background.raise_()
        self.Title.raise_()
        self.aRecyclable.raise_()
        self.bHarmful.raise_()
        self.cKitchen.raise_()
        self.dOther.raise_()
        self.back_Warn.raise_()
        self.WarnRecyclable.raise_()
        self.WarnHarmful.raise_()
        self.WarnKitchen.raise_()
        self.WarnOther.raise_()
        self.ifWarnR.raise_()
        self.ifWarnH.raise_()
        self.ifWarnK.raise_()
        self.ifWarnO.raise_()
        self.Information.raise_()
        self.LOGO.raise_()
        self.videoWidget.raise_()
        self.Camera.raise_()

        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile("/home/dhu/TrashDetector/icons/Trash.mp4")))
        self.mediaPlayer.play()

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def restart(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.mediaPlayer.setPosition(0)
            self.mediaPlayer.play()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "智能垃圾桶"))
        self.Title.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:24pt;\">中国大学生工程实践与创新能力大赛</span></p><p align=\"center\"><span style=\" font-size:20pt; font-weight:600;\">东华大学-环卫大队</span></p></body></html>"))
        self.WarnRecyclable.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt;\">可回收垃圾：</span></p></body></html>"))
        self.WarnHarmful.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt;\">有害垃圾：</span></p></body></html>"))
        self.WarnKitchen.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt;\">厨余垃圾：</span></p></body></html>"))
        self.WarnOther.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt;\">其他垃圾：</span></p></body></html>"))
        self.ifWarnR.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt;\">未满</span></p></body></html>"))
        self.ifWarnH.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt;\">未满</span></p></body></html>"))
        self.ifWarnK.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt;\">未满</span></p></body></html>"))
        self.ifWarnO.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt;\">未满</span></p></body></html>"))

    def warn_R(self, w):
        _translate = QtCore.QCoreApplication.translate
        if w == False:
            self.ifWarnR.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt;\">未满</span></p></body></html>"))
        if w == True:
            self.ifWarnR.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt;\">满载</span></p></body></html>"))
    def warn_H(self, w):
        _translate = QtCore.QCoreApplication.translate
        if w == False:
            self.ifWarnH.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt;\">未满</span></p></body></html>"))
        if w == True:
            self.ifWarnH.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt;\">满载</span></p></body></html>"))
    def warn_K(self, w):
        _translate = QtCore.QCoreApplication.translate
        if w == False:
            self.ifWarnK.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt;\">未满</span></p></body></html>"))
        if w == True:
            self.ifWarnK.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt;\">满载</span></p></body></html>"))
    def warn_O(self, w):
        _translate = QtCore.QCoreApplication.translate
        if w == False:
            self.ifWarnO.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt;\">未满</span></p></body></html>"))
        if w == True:
            self.ifWarnO.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt;\">满载</span></p></body></html>"))

    def add_text(self, t):
        new_text = self.Information.toPlainText() + t
        self.Information.setPlainText(new_text)
        self.Information.moveCursor(QTextCursor.End)
    def back_text(self):
        text = self.Information.toPlainText()
        if len(text) > 6:
            self.Information.setPlainText(text[:-6])

    def change_num(self, val):
        self.anum = val[0]
        self.bnum = val[1]
        self.cnum = val[2]
        self.dnum = val[3]
        self.aRecyclable.setProperty("value", self.anum)
        self.bHarmful.setProperty("value", self.bnum)
        self.cKitchen.setProperty("value", self.cnum)
        self.dOther.setProperty("value", self.dnum)
    
    def display_image(self, img):
        # Convert the Mat image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert the RGB image to QImage
        qimg = QtGui.QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.Camera.setPixmap(pixmap)
        
ui = Ui_MainWindow()


# 串口通信进程
# buf_send: 0:识别到的垃圾种类 1:null 2:null 3:null
# buf_receive: 0:处理的垃圾种类 1:可回收垃圾是否满载 2:有害垃圾是否满载 3:厨余垃圾是否满载 4:其他垃圾是否满载 
def ser_com():
    shm_send = shared_memory.SharedMemory(name='ser_send')
    shm_receive = shared_memory.SharedMemory(name='ser_receive')
    buf_send = shm_send.buf
    buf_receive = shm_receive.buf
    # 初始化不为空，防止读取空串口导致后续处理长度溢出报错
    resstr = "*"
    while True:
        # 反复尝试连接串口，类似与热插拔
        while True:
            try:
                # os.system("echo password | sudo -S chmod 777 /dev/ttyUSB0")
                ser = serial.Serial('/dev/ttyUSB0', 115200, 8, 'N', 1)
                break
            except Exception as e:
                print("Serial Wrong!")
                time.sleep(1)

        infm = -1
        for i in range(4):
            if buf_send[i] == 1:
                infm = i
                buf_send[:4] = bytearray([0,0,0,0])

        # 若串口断开，则直接跳至尝试连接串口处
        try:
            # print(infm)
            ser.write('a{0}b{1}c{1}d{1}e'.format(str(infm), str(inf)).encode())
            resstr = ser.readline().decode()
        except Exception as e:
            ser.close()
            continue

        # 若读取空串口，则赋值为非空
        if len(resstr) == 0:
            resstr = "*"
        # print(resstr)

        # 过滤掉错误信息
        if resstr[0] == 'a':
            i = 1
            cod1 = ''
            while resstr[i] != 'b':
                cod1 += resstr[i]
                i += 1
            i += 1
            cod2 = ''
            while resstr[i] != 'c':
                cod2 += resstr[i]
                i += 1
            i += 1
            cod3 = ''
            while resstr[i] != 'd':
                cod3 += resstr[i]
                i += 1
            i += 1
            cod4 = ''
            while resstr[i] != 'e':
                cod4 += resstr[i]
                i += 1
            i += 1
            cod5 = ''
            while resstr[i] != 'f':
                cod5 += resstr[i]
                i += 1
            
            buf_receive[0] = int(cod1)
            buf_receive[1] = int(cod2)
            buf_receive[2] = int(cod3)
            buf_receive[3] = int(cod4)
            buf_receive[4] = int(cod5)
        # print(buf_receive)
        # print(infm)
        ser.close()


# 0, 1, 2, 3:可回收垃圾, 有害垃圾, 厨余垃圾, 其他垃圾
@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    shm_send = shared_memory.SharedMemory(name='ser_send')
    buf_send = shm_send.buf

    Recyclable = 0
    Harmful = 0
    Kitchen = 0
    Other = 0
    num=0
    ui_num = 0

    # 滤波，防止偶尔出现的误识别
    lenlist = 15
    threshold = 9
    Trash_list = []
    Trash_num = []
    warn_list = []

    source = str(source)
    webcam = source.isnumeric() or source.endswith('.streams')
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        while True:
            try:
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
                break
            except Exception as e:
                print("Camera Wrong!")
                time.sleep(1)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    dt = (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset: # 每一帧
        # 处理串口信息，UI显示
        shm_receive=shared_memory.SharedMemory(name='ser_receive')
        buf_receive=shm_receive.buf
        # print("buf_receive buf_receive buf_receive buf_receive")
        # print("{0},{1},{2},{3},{4}".format(buf_receive[0], buf_receive[1], buf_receive[2], buf_receive[3], buf_receive[4]))
        if buf_receive[0] != 5 and buf_receive[0] != 4:
            num += 1
        if buf_receive[0] == 0:
            ui.add_text("{0} 可回收垃圾 {1} 处理中...".format(num, ui_num))
            Recyclable += ui_num
            ui.change_num([Recyclable, Harmful, Kitchen, Other])
        if buf_receive[0] == 1:
            ui.add_text("{0} 有害垃圾 {1} 处理中...".format(num, ui_num))
            Harmful += ui_num
            ui.change_num([Recyclable, Harmful, Kitchen, Other])
        if buf_receive[0] == 2:
            ui.add_text("{0} 厨余垃圾 {1} 处理中...".format(num, ui_num))
            Kitchen += ui_num
            ui.change_num([Recyclable, Harmful, Kitchen, Other])
        if buf_receive[0] == 3:
            ui.add_text("{0} 其他垃圾 {1} 处理中...".format(num, ui_num))
            Other += ui_num
            ui.change_num([Recyclable, Harmful, Kitchen, Other])
        if buf_receive[0] == 4:
            ui.back_text()
            ui.add_text("分类完成！\n")
        Wr, Wh, Wk, Wo = 0, 0, 0, 0
        if buf_receive[1] == 1:
            Wr = 1
        if buf_receive[2] == 1:
            Wh = 1
        if buf_receive[3] == 1:
            Wk = 1
        if buf_receive[4] == 1:
            Wo = 1
        warn_list.append([Wr, Wh, Wk, Wo])
        buf_receive[0] = 5

        # 确认垃圾桶满载
        Wr, Wh, Wk, Wo = 0, 0, 0, 0
        if len(warn_list) >= lenlist:
            Wr, Wh, Wk, Wo = 0, 0, 0, 0
            count = 0
            for i in range(lenlist):
                Wr += warn_list[i][0]
                Wh += warn_list[i][1]
                Wk += warn_list[i][2]
                Wo += warn_list[i][3]
            if Wr >= threshold:
                ui.warn_R(True)
            else:
                ui.warn_R(False)
            if Wh >= threshold:
                ui.warn_H(True)
            else:
                ui.warn_H(False)
            if Wk >= threshold:
                ui.warn_K(True)
            else:
                ui.warn_K(False)    
            if Wo >= threshold:
                ui.warn_O(True)
            else:
                ui.warn_O(False)
            warn_list.pop(0)

        # 确认垃圾的存在
        Tr, Th, Tk, To = 0, 0, 0, 0
        if len(Trash_list) >= lenlist:
            Tr, Th, Tk, To = 0, 0, 0, 0
            count = 0
            for i in range(lenlist):
                Tr += Trash_list[i][0]
                Th += Trash_list[i][1]
                Tk += Trash_list[i][2]
                To += Trash_list[i][3]
            if Tr >= threshold:
                count += 1
            if Th >= threshold:
                count += 1
            if Tk >= threshold:
                count += 1
            if To >= threshold:
                count += 1
            Trash_list.pop(0)
            if count == 1:
                if Tr >= threshold:
                    infm = 0
                    buf_send[infm] = 1
                    ui_num = Trash_num[0]
                if Th >= threshold:
                    infm = 1
                    buf_send[infm] = 1
                    ui_num = Trash_num[1]
                if Tk >= threshold:
                    infm = 2
                    buf_send[infm] = 1
                    ui_num = Trash_num[2]
                if To >= threshold:
                    infm = 3
                    buf_send[infm] = 1
                    ui_num = Trash_num[3]
                Trash_list.clear()

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        # Inference
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # 识别到目标
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # 记录每类个数
                now_list = []
                Tr, Th, Tk, To = 0, 0, 0, 0
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    if c == 0:
                        Tr = n.item()
                    if c == 1:
                        Th = n.item()
                    if c == 2:
                        Tk = n.item()
                    if c == 3:
                        To = n.item()
                if Tr:
                    now_list.append(1)
                else:
                    now_list.append(0)
                if Th:
                    now_list.append(1)
                else:
                    now_list.append(0)
                if Tk:
                    now_list.append(1)
                else:
                    now_list.append(0)
                if To:
                    now_list.append(1)
                else:
                    now_list.append(0)
                Trash_list.append(now_list)
                Trash_num.clear()
                Trash_num.append(Tr)
                Trash_num.append(Th)
                Trash_num.append(Tk)
                Trash_num.append(To)
                # 判断是否存在多类物体
                count = 0
                if Tr > 0:
                    count += 1
                if Th > 0:
                    count += 1
                if Tk > 0:
                    count += 1
                if To > 0:
                    count += 1
                if count > 1:
                    Trash_list.clear()
                    time.sleep(1)
                # 标框
                for *xyxy, conf, cls in reversed(det):
                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
            # 未识别到目标
            else:
                Trash_list.append([0, 0, 0, 0])

            # 显示图像
            im0 = annotator.result()
            if view_img:
                cv2.imshow('test', np.zeros((1)))
                cv2.destroyAllWindows()
                ui.display_image(im0)
                cv2.waitKey(1)

        # 显示识别信息
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        # LOGGER.info(f"{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'models/Trash_1020.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.85, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def detect():
    opt = parse_opt()
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


def UI():
    app = QApplication(sys.argv)
    
    mw = QMainWindow()

    global ui
    ui.setupUi(mw)
    mw.show()
    detect()
    
    sys.exit(app.exec_())
