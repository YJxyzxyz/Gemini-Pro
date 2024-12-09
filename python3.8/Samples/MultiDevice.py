from ObTypes import *
from Property import *
import Pipeline
import StreamProfile
from Error import ObException
import cv2
import numpy as np
import open3d as o3d
import sys
import os

# 设置保存图像的文件夹路径
save_directory = "saved_images"
os.makedirs(save_directory, exist_ok=True)

q = 113
ESC = 27
s = 115  # ASCII for 's'


def create_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    point_cloud.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.0)
    return point_cloud


try:
    pipe = Pipeline.Pipeline(None, None)
    config = Pipeline.Config()

    try:
        profiles = pipe.getStreamProfileList(OB_PY_SENSOR_COLOR)
        videoProfile = profiles.getProfile(0)
        colorProfile = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
        colorWidth = colorProfile.width()
        colorHeight = colorProfile.height()
        config.enableStream(colorProfile)
    except ObException as e:
        print("Color sensor error: %s" % e.getMessage())
        sys.exit()

    try:
        profiles = pipe.getStreamProfileList(OB_PY_SENSOR_DEPTH)
        videoProfile = profiles.getProfile(0)
        depthProfile = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
        depthWidth = depthProfile.width()
        depthHeight = depthProfile.height()
        config.enableStream(depthProfile)
    except ObException as e:
        print("Depth sensor error: %s" % e.getMessage())
        sys.exit()

    pipe.start(config, None)

    # 设置镜像属性
    if pipe.getDevice().isPropertySupported(OB_PY_PROP_COLOR_MIRROR_BOOL, OB_PY_PERMISSION_WRITE):
        pipe.getDevice().setBoolProperty(OB_PY_PROP_COLOR_MIRROR_BOOL, True)

    while True:
        frameSet = pipe.waitForFrames(100)
        if frameSet is None:
            continue

        colorFrame = frameSet.colorFrame()
        depthFrame = frameSet.depthFrame()

        if colorFrame is not None and depthFrame is not None:
            colorData = np.asarray(colorFrame.data())
            depthData = np.asarray(depthFrame.data())

            # 处理颜色数据
            if colorFrame.format() == OB_PY_FORMAT_MJPG:
                colorMat = cv2.imdecode(colorData, 1)
                colorMat = cv2.resize(colorMat, (depthWidth, depthHeight))
            elif colorFrame.format() == OB_PY_FORMAT_RGB888:
                colorMat = colorData.reshape((colorHeight, colorWidth, 3))
                colorMat = cv2.cvtColor(colorMat, cv2.COLOR_RGB2BGR)
                colorMat = cv2.resize(colorMat, (depthWidth, depthHeight))

            # 处理深度数据
            depthMat = depthData.reshape((depthHeight, depthWidth, 2))
            depthMat = depthMat[:, :, 0] + depthMat[:, :, 1] * 256
            depthMat = (depthMat * depthFrame.getValueScale()).astype('uint16')

            # 水平翻转深度图像
            depthMat = cv2.flip(depthMat, 1)

            # 显示图像（仅用于查看效果）
            depthMatDisp = cv2.applyColorMap(cv2.convertScaleAbs(depthMat, alpha=0.03), cv2.COLORMAP_JET)
            combinedMat = np.hstack((colorMat, depthMatDisp))
            cv2.imshow("Color and Depth Viewer", combinedMat)

            key = cv2.waitKey(1)
            if key == ESC or key == q:
                break
            elif key == s:
                # 保存RGB和深度图
                color_image_path = os.path.join(save_directory, "color_image.png")
                depth_image_path = os.path.join(save_directory, "depth_image.png")

                cv2.imwrite(color_image_path, colorMat)
                cv2.imwrite(depth_image_path, depthMat)  # 保存单通道深度图

                print(f"Saved color image to {color_image_path}")
                print(f"Saved depth image to {depth_image_path}")

    cv2.destroyAllWindows()
    pipe.stop()

except ObException as e:
    print("Error: %s" % e.getMessage())
