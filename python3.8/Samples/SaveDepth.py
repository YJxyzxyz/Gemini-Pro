from ObTypes import *
from Property import *
import Pipeline
import StreamProfile
from Error import ObException
import cv2
import numpy as np
import sys
import os

q = 113
ESC = 27
s = 115

# 指定保存深度图的文件夹路径
save_folder = "./depth_images"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

try:
    # 创建一个Pipeline对象，这是整个高级API的入口点，可以通过Pipeline轻松打开和关闭流
    pipe = Pipeline.Pipeline(None, None)
    # 通过创建Config对象来配置Pipeline中要启用或禁用的流
    config = Pipeline.Config()

    windowsWidth = 0
    windowsHeight = 0
    try:
        # 获取深度相机的所有流配置，包括流分辨率、帧率和帧格式
        profiles = pipe.getStreamProfileList(OB_PY_SENSOR_DEPTH)

        videoProfile = None
        try:
            # 选择默认分辨率来打开流，可以通过配置文件配置默认分辨率
            videoProfile = profiles.getProfile(0)
        except ObException as e:
            print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
            e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))

        depthProfile = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
        windowsWidth = depthProfile.width()
        windowsHeight = depthProfile.height()
        config.enableStream(depthProfile)
    except ObException as e:
        print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
        e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
        print("当前设备不支持深度传感器！")
        sys.exit()

    # 启动在Config中配置的流，如果没有传递参数，将启动默认配置的流
    pipe.start(config, None)

    # 检查是否支持镜像属性并具有写权限
    if pipe.getDevice().isPropertySupported(OB_PY_PROP_DEPTH_MIRROR_BOOL, OB_PY_PERMISSION_WRITE):
        # 设置镜像
        pipe.getDevice().setBoolProperty(OB_PY_PROP_DEPTH_MIRROR_BOOL, True)

    frame_count = 0
    while True:
        # 以阻塞方式等待一帧数据，这是一个包含所有已启用流帧数据的复合帧，并设置帧等待超时为100ms
        frameSet = pipe.waitForFrames(100)
        if frameSet == None:
            continue
        else:
            # 在窗口中渲染一帧数据，这里只渲染深度帧
            depthFrame = frameSet.depthFrame()
            if depthFrame != None:
                size = depthFrame.dataSize()
                data = depthFrame.data()
                if size != 0:
                    # 将帧数据调整为(height, width, 2)
                    data = np.resize(data, (windowsHeight, windowsWidth, 2))

                    # 将帧数据从8位转换为16位
                    newData = data[:, :, 0] + data[:, :, 1] * 256
                    # 将帧数据转换为1mm单位
                    newData = (newData * depthFrame.getValueScale()).astype('uint16')
                    # 渲染显示
                    newData = (newData / 32).astype('uint8')
                    # 将帧数据从灰度转换为RGB
                    newData = cv2.cvtColor(newData, cv2.COLOR_GRAY2RGB)

                    cv2.namedWindow("DepthViewer", cv2.WINDOW_NORMAL)

                    cv2.imshow("DepthViewer", newData)

                    key = cv2.waitKey(1)
                    if key == ESC or key == q:
                        cv2.destroyAllWindows()
                        break
                    elif key == s:
                        # 保存深度图到指定文件夹
                        save_path = os.path.join(save_folder, f"depth_frame_{frame_count}.png")
                        cv2.imwrite(save_path, newData)
                        print(f"Depth image saved to {save_path}")
                        frame_count += 1
    pipe.stop()

except ObException as e:
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
    e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
