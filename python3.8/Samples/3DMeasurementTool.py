import open3d as o3d
import numpy as np
import cv2
from ObTypes import *
import Pipeline
import Context
from Error import ObException
import sys


def create_point_cloud_from_frames(color_frame, depth_frame, camera_intrinsics, depth_scale):
    """从颜色帧和深度帧创建Open3D点云对象"""
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    o3d_color_image = o3d.geometry.Image(color_image_rgb)
    o3d_depth_image = o3d.geometry.Image(depth_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color_image,
        o3d_depth_image,
        depth_scale=1.0 / depth_scale,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False)

    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics)

    return point_cloud


def pick_points(pcd):
    """点云拾取回调函数"""
    print("\n1) 按住 [SHIFT] + 鼠标左键 点击来选择点。")
    print("2) 选择两个点后，将在终端显示距离。")
    print("3) 按 [ESC] 键退出。")

    # 创建一个支持点选的可视化窗口
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="三维测量工具", width=800, height=600)
    vis.add_geometry(pcd)

    # 运行可视化窗口，等待用户拾取点
    vis.run()
    vis.destroy_window()

    # 获取被选中的点的索引
    picked_indices = vis.get_picked_points()

    # 如果用户选择了至少两个点
    if len(picked_indices) >= 2:
        # 获取前两个被选中的点的坐标
        p1_index = picked_indices[0]
        p2_index = picked_indices[1]

        p1_coord = np.asarray(pcd.points)[p1_index]
        p2_coord = np.asarray(pcd.points)[p2_index]

        # 计算两点之间的欧式距离
        distance = np.linalg.norm(p1_coord - p2_coord)

        print("\n========================================")
        print(f"第一个点的坐标: {p1_coord}")
        print(f"第二个点的坐标: {p2_coord}")
        print(f"两点之间的距离是: {distance:.4f} 米")
        print("========================================")
    else:
        print("\n没有选择足够的点来进行测量。")


def main():
    try:
        # 初始化Orbbec SDK
        ctx = Context.Context(None)
        ctx.setLoggerSeverity(OB_PY_LOG_SEVERITY_ERROR)
        pipeline = Pipeline.Pipeline(None, None)
        config = Pipeline.Config()

        # 使能深度和彩色数据流
        try:
            depth_profiles = pipeline.getStreamProfileList(OB_PY_SENSOR_DEPTH)
            depth_profile = depth_profiles.getProfile(0)
            config.enableStream(depth_profile)
        except ObException as e:
            print(f"无法使能深度流: {e}")
            return

        try:
            color_profiles = pipeline.getStreamProfileList(OB_PY_SENSOR_COLOR)
            color_profile = color_profiles.getProfile(0)
            config.enableStream(color_profile)
        except ObException as e:
            print(f"无法使能彩色流: {e}")
            return

        config.setAlignMode(OB_PY_ALIGN_D2C_HW_MODE)
        pipeline.start(config, None)

        # 获取相机内参
        camera_param = pipeline.getCameraParam()
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=color_profile.width(),
            height=color_profile.height(),
            fx=camera_param.rgbIntrinsic.fx,
            fy=camera_param.rgbIntrinsic.fy,
            cx=camera_param.rgbIntrinsic.cx,
            cy=camera_param.rgbIntrinsic.cy
        )

        print("正在捕获场景，请稍候...")

        # 捕获一帧稳定的图像
        frameset = None
        for _ in range(10):  # 尝试捕获10次，以获得稳定图像
            frameset = pipeline.waitForFrames(200)
            if frameset is not None and frameset.colorFrame() is not None and frameset.depthFrame() is not None:
                break

        if frameset is None:
            print("捕获图像失败，请检查相机连接。")
            pipeline.stop()
            return

        color_frame = frameset.colorFrame()
        depth_frame = frameset.depthFrame()
        depth_scale = depth_frame.getValueScale()

        # 从帧数据创建点云
        pcd = create_point_cloud_from_frames(color_frame, depth_frame, intrinsics, depth_scale)

        # 移除统计学上的离群点，使点云更干净
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_cleaned = pcd.select_by_index(ind)

        print("场景捕获完成，正在打开测量工具...")

        # 停止相机数据流
        pipeline.stop()

        # 调用点选和测量函数
        pick_points(pcd_cleaned)

    except ObException as e:
        print(f"SDK 异常: {e}")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()