import torch
import cv2
import numpy as np
import open3d as o3d
from ObTypes import *
import Pipeline
import Context
from Error import ObException
import random


def create_point_cloud_from_frames(color_frame, depth_frame, camera_intrinsics, depth_scale):
    """从对齐的颜色帧和深度帧创建Open3D点云"""
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # 将BGR图像转换为RGB
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    o3d_color_image = o3d.geometry.Image(color_image_rgb)
    o3d_depth_image = o3d.geometry.Image(depth_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color_image,
        o3d_depth_image,
        depth_scale=1.0 / depth_scale,
        depth_trunc=5.0,  # 截断大于5米的深度值
        convert_rgb_to_intensity=False)

    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics)

    return point_cloud


def main():
    try:
        # --- 1. 加载YOLOv5模型 ---
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # 从本地加载模型可以避免网络问题
        model = torch.hub.load('./yolov5', 'yolov5s', source='local', pretrained=True)

        # --- 2. 初始化Orbbec SDK ---
        ctx = Context.Context(None)
        ctx.setLoggerSeverity(OB_PY_LOG_SEVERITY_ERROR)
        pipeline = Pipeline.Pipeline(None, None)
        config = Pipeline.Config()

        # 使能深度和彩色数据流
        try:
            color_profiles = pipeline.getStreamProfileList(OB_PY_SENSOR_COLOR)
            color_profile = color_profiles.getVideoStreamProfile(640, 480, OB_PY_FORMAT_BGR, 30)
            config.enableStream(color_profile)
        except ObException as e:
            print(f"无法使能彩色流: {e}")
            return

        try:
            depth_profiles = pipeline.getStreamProfileList(OB_PY_SENSOR_DEPTH)
            depth_profile = depth_profiles.getProfile(0)  # 选择与彩色流最匹配的深度流
            config.enableStream(depth_profile)
        except ObException as e:
            print(f"无法使能深度流: {e}")
            return

        # 确保D2C对齐
        config.setAlignMode(OB_PY_ALIGN_D2C_HW_MODE)
        pipeline.start(config, None)

        # --- 3. 获取相机内参并初始化Open3D ---
        camera_param = pipeline.getCameraParam()
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=color_profile.width(),
            height=color_profile.height(),
            fx=camera_param.rgbIntrinsic.fx,
            fy=camera_param.rgbIntrinsic.fy,
            cx=camera_param.rgbIntrinsic.cx,
            cy=camera_param.rgbIntrinsic.cy
        )

        vis = o3d.visualization.Visualizer()
        vis.create_window("Segmented 3D Objects", 800, 600)

        print("按 'Q' 键退出...")

        # 存储当前帧检测到的物体点云
        geometries = {}

        while True:
            # --- 4. 捕获帧数据 ---
            frameset = pipeline.waitForFrames(100)
            if frameset is None:
                continue
            color_frame = frameset.colorFrame()
            depth_frame = frameset.depthFrame()
            if color_frame is None or depth_frame is None:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # --- 5. 使用YOLO进行物体检测 ---
            results = model(frame)
            detections = results.pandas().xyxy[0]  # 获取检测结果

            # --- 6. 3D分割与可视化 ---
            # 首先移除上一帧的所有几何体
            for geo_id in geometries:
                vis.remove_geometry(geometries[geo_id], reset_bounding_box=False)
            geometries.clear()

            # 创建当前帧的完整点云
            full_pcd = create_point_cloud_from_frames(color_frame, depth_frame, intrinsics, depth_frame.getValueScale())
            full_pcd_points = np.asarray(full_pcd.points)
            full_pcd_colors = np.asarray(full_pcd.colors)

            for index, row in detections.iterrows():
                # 获取边界框坐标
                xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                confidence = row['confidence']
                label = f"{row['name']} {confidence:.2f}"

                if confidence > 0.5:  # 只处理置信度大于0.5的物体
                    # --- 2D 可视化 ---
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # --- 3D 分割 ---
                    # 创建一个掩码来选择位于边界框内的点
                    # 注意：点云是一维数组，需要映射回2D图像坐标
                    u = (full_pcd_points[:, 0] * intrinsics.fx / full_pcd_points[:, 2] + intrinsics.cx).astype(int)
                    v = (full_pcd_points[:, 1] * intrinsics.fy / full_pcd_points[:, 2] + intrinsics.cy).astype(int)

                    mask = (u >= xmin) & (u < xmax) & (v >= ymin) & (v < ymax)

                    segmented_points = full_pcd_points[mask]
                    segmented_colors = full_pcd_colors[mask]

                    if len(segmented_points) > 0:
                        segmented_pcd = o3d.geometry.PointCloud()
                        segmented_pcd.points = o3d.utility.Vector3dVector(segmented_points)
                        # 为不同的物体分配随机颜色以便区分
                        random_color = [random.random() for _ in range(3)]
                        segmented_pcd.paint_uniform_color(random_color)

                        # 添加到可视化窗口
                        geometries[index] = segmented_pcd
                        vis.add_geometry(segmented_pcd, reset_bounding_box=False)

            # 显示2D结果
            cv2.imshow("Object Detection (2D)", frame)

            # 更新3D渲染
            vis.poll_events()
            vis.update_renderer()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except ObException as e:
        print(f"SDK 异常: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if 'pipeline' in locals():
            pipeline.stop()
        if 'vis' in locals():
            vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()