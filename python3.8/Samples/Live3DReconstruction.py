import open3d as o3d
import numpy as np
import cv2
from ObTypes import *
import Pipeline
import Context
from Error import ObException
import sys

# 点云下采样体素大小，值越大，点云越稀疏，但处理速度越快
VOXEL_SIZE = 0.02
# ICP 配准的最大对应点距离
MAX_CORRESPONDENCE_DISTANCE = 0.05


def create_point_cloud_from_frames(color_frame, depth_frame, camera_intrinsics, depth_scale):
    """从颜色帧和深度帧创建Open3D点云对象"""

    # 将帧数据转换为numpy数组
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # 将BGR图像转换为RGB
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # 创建Open3D图像对象
    o3d_color_image = o3d.geometry.Image(color_image_rgb)
    o3d_depth_image = o3d.geometry.Image(depth_image)

    # 从RGB-D图像创建点云
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color_image,
        o3d_depth_image,
        depth_scale=1.0 / depth_scale,  # 深度值单位转换
        depth_trunc=3.0,  # 截断大于3米的深度值
        convert_rgb_to_intensity=False)

    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics)

    return point_cloud


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

        # 设置对齐模式为D2C硬件对齐
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

        # 初始化Open3D可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window("实时三维重建")

        # 全局点云模型
        global_pcd = o3d.geometry.PointCloud()
        # 用于存储上一帧点云
        previous_pcd = None
        # 是否为第一帧
        is_first_frame = True

        print("按 'ESC' 键退出...")

        while True:
            # 等待帧数据
            frameset = pipeline.waitForFrames(100)
            if frameset is None:
                continue

            color_frame = frameset.colorFrame()
            depth_frame = frameset.depthFrame()
            if color_frame is None or depth_frame is None:
                continue

            depth_scale = depth_frame.getValueScale()

            # 从帧数据创建点云
            source_pcd = create_point_cloud_from_frames(color_frame, depth_frame, intrinsics, depth_scale)

            # 对点云进行预处理：下采样
            source_pcd_down = source_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

            if is_first_frame:
                global_pcd = source_pcd_down
                previous_pcd = source_pcd_down
                vis.add_geometry(global_pcd)
                is_first_frame = False
            else:
                # 使用ICP算法进行点云配准
                # target是上一帧的点云，source是当前帧的点云
                # 目的是计算从source到target的变换矩阵
                transformation_icp = o3d.pipelines.registration.registration_icp(
                    source_pcd_down, previous_pcd, MAX_CORRESPONDENCE_DISTANCE,
                    np.identity(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

                # 将当前帧点云变换到全局坐标系下
                source_pcd_down.transform(transformation_icp.transformation)

                # 将变换后的点云合并到全局点云中
                global_pcd += source_pcd_down
                # 对合并后的全局点云再次下采样，防止点云过于密集
                global_pcd = global_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

                # 更新上一帧点云为当前帧点云
                previous_pcd = source_pcd_down

                # 更新可视化窗口中的几何体
                vis.update_geometry(global_pcd)

            # 更新渲染并处理窗口事件
            if not vis.poll_events():
                break
            vis.update_renderer()

    except ObException as e:
        print(f"SDK 异常: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if 'pipeline' in locals():
            pipeline.stop()
        if 'vis' in locals():
            vis.destroy_window()


if __name__ == "__main__":
    main()