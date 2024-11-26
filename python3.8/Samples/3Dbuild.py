from ObTypes import *
from Property import *
import Pipeline
import StreamProfile
from Error import ObException
import cv2
import numpy as np
import open3d as o3d  # 用于处理点云的库
import sys

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

            # 显示图像
            depthMatDisp = cv2.applyColorMap(cv2.convertScaleAbs(depthMat, alpha=0.03), cv2.COLORMAP_JET)
            combinedMat = np.hstack((colorMat, depthMatDisp))
            cv2.imshow("Color and Depth Viewer", combinedMat)

            key = cv2.waitKey(1)
            if key == ESC or key == q:
                break
            elif key == s:
                # 生成点云图
                print("Generating point cloud for 3D reconstruction...")
                points = np.zeros((depthHeight * depthWidth, 6), dtype=np.float32)
                points[:, 0] = np.repeat(np.arange(depthWidth), depthHeight)
                points[:, 1] = np.tile(np.arange(depthHeight), depthWidth)
                points[:, 2] = depthMat.flatten()
                points[:, 3:] = colorMat.reshape(-1, 3)

                point_cloud = create_point_cloud(points)

                # 使用open3d进行三维重建（例如Poisson重建）
                # 进行去噪和下采样
                point_cloud = point_cloud.voxel_down_sample(voxel_size=0.02)
                cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                point_cloud = point_cloud.select_by_index(ind)

                # 法向量估计
                point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

                # 三维重建（Poisson重建）
                print("Performing Poisson reconstruction...")
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)

                # 可视化重建结果
                mesh.compute_vertex_normals()
                o3d.visualization.draw_geometries([mesh])
                print("3D reconstruction visualization complete.")

    cv2.destroyAllWindows()
    pipe.stop()

except ObException as e:
    print("Error: %s" % e.getMessage())