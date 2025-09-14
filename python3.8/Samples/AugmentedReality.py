import open3d as o3d
import numpy as np
import cv2
import time
from ObTypes import *
import Pipeline
import Context
from Error import ObException


# --- 全局状态变量 ---
class AppState:
    def __init__(self):
        self.state = "DETECTING_PLANE"  # 初始状态：检测平面
        self.plane_model = None  # 检测到的平面模型
        self.plane_mesh = None  # 用于可视化平面的网格
        self.virtual_object = None  # 虚拟物体（龙模型）
        self.object_transform = np.identity(4)  # 虚拟物体的位置姿态
        self.previous_pcd = None  # 上一帧的点云，用于追踪
        self.camera_pose = np.identity(4)  # 相机位姿


app_state = AppState()


def create_point_cloud_from_frames(color_frame, depth_frame, intrinsics, depth_scale):
    """从帧数据创建Open3D点云"""
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    o3d_color = o3d.geometry.Image(color_image_rgb)
    o3d_depth = o3d.geometry.Image(depth_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth, depth_scale=1.0 / depth_scale, depth_trunc=4.0, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    return pcd


def detect_plane_and_place_object(vis):
    """检测平面并等待用户点击放置物体"""
    print("AR系统启动：正在检测平面...")

    # 切换到平面检测状态
    app_state.state = "DETECTING_PLANE"

    # 获取当前视图中的点云
    view_pcd = vis.get_view_control().convert_to_pinhole_camera_parameters().intrinsic

    # 我们需要从主循环中获取点云，这里只是触发状态改变
    # 实际的检测将在主循环中进行
    return False


def on_mouse_click(vis, action, mods):
    """鼠标点击回调函数"""
    if action == 1 and app_state.state == "AWAITING_PLACEMENT":  # 1 代表按下
        # 获取屏幕坐标
        # Open3D没有直接获取鼠标点击3D坐标的API，我们用一个技巧：
        # 假设用户点击屏幕中心来放置物体
        print("检测到放置指令（模拟点击中心）...")

        cx = vis.get_view_control().get_camera_parameters().intrinsic.get_principal_point()[0]
        cy = vis.get_view_control().get_camera_parameters().intrinsic.get_principal_point()[1]

        # 从深度图中获取中心点的深度
        # 这是一个简化，理想情况需要更复杂的 unproject
        # 我们直接使用平面模型的中心点作为放置点
        plane_center = app_state.plane_mesh.get_center()

        print(f"物体将放置在: {plane_center}")

        # 将虚拟物体移动到该位置
        app_state.object_transform[0:3, 3] = plane_center
        app_state.virtual_object.transform(app_state.object_transform)

        # 更新状态为追踪模式
        app_state.state = "TRACKING"
        print("状态切换 -> TRACKING. 您现在可以移动相机了。")

        # 移除平面指示器并添加物体
        vis.remove_geometry(app_state.plane_mesh, reset_bounding_box=False)
        vis.add_geometry(app_state.virtual_object, reset_bounding_box=False)


def main():
    try:
        # --- 1. 初始化相机和SDK ---
        ctx = Context.Context(None)
        pipeline = Pipeline.Pipeline(None, None)
        config = Pipeline.Config()

        try:
            color_profiles = pipeline.getStreamProfileList(OB_PY_SENSOR_COLOR)
            color_profile = color_profiles.getVideoStreamProfile(640, 480, OB_PY_FORMAT_BGR, 30)
            config.enableStream(color_profile)
        except ObException as e:
            return

        try:
            depth_profiles = pipeline.getStreamProfileList(OB_PY_SENSOR_DEPTH)
            depth_profile = depth_profiles.getProfile(0)
            config.enableStream(depth_profile)
        except ObException as e:
            return

        config.setAlignMode(OB_PY_ALIGN_D2C_HW_MODE)
        pipeline.start(config, None)

        camera_param = pipeline.getCameraParam()
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            color_profile.width(), color_profile.height(),
            camera_param.rgbIntrinsic.fx, camera_param.rgbIntrinsic.fy,
            camera_param.rgbIntrinsic.cx, camera_param.rgbIntrinsic.cy)

        # --- 2. 加载虚拟物体模型 ---
        try:
            print("正在加载虚拟模型 'Fantasy Dragon.ply'...")
            # 请确保路径正确！
            dragon_path = "./RGBPoints/Fantasy Dragon.ply"
            app_state.virtual_object = o3d.io.read_point_cloud(dragon_path)
            # 缩小模型并预设一个颜色
            app_state.virtual_object.scale(0.3, center=app_state.virtual_object.get_center())
            app_state.virtual_object.paint_uniform_color([0.9, 0.1, 0.1])  # 红色
        except Exception as e:
            print(f"错误: 无法加载模型 {dragon_path}。请确保文件存在。")
            print(e)
            return

        # --- 3. 初始化Open3D可视化 ---
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("Augmented Reality", 1280, 720)

        # 注册按键回调：按'P'键开始平面检测和放置
        vis.register_key_callback(ord("P"), detect_plane_and_place_object)

        # 模拟鼠标点击放置，因为获取3D坐标比较复杂
        # vis.register_mouse_action_callback(on_mouse_click)

        print("\n" + "=" * 50)
        print("增强现实原型程序")
        print("操作指南:")
        print("  1. 将相机对准一个平坦的表面 (如桌面或地面)。")
        print("  2. 在窗口激活时，按下 'P' 键。")
        print("  3. 程序会自动检测平面并用半透明绿色高亮显示。")
        print("  4. 再次按下 'P' 键确认放置虚拟物体。")
        print("  5. 放置后，您可以缓慢移动相机来观察。")
        print("  6. 按 'Q' 或关闭窗口退出。")
        print("=" * 50 + "\n")

        pcd = o3d.geometry.PointCloud()
        is_first_frame = True

        while True:
            # --- 4. 主循环 ---
            frameset = pipeline.waitForFrames(100)
            if frameset is None: continue
            color_frame = frameset.colorFrame()
            depth_frame = frameset.depthFrame()
            if color_frame is None or depth_frame is None: continue

            current_pcd = create_point_cloud_from_frames(color_frame, depth_frame, intrinsics,
                                                         depth_frame.getValueScale())
            current_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # 翻转点云以匹配视图

            if is_first_frame:
                pcd.points = current_pcd.points
                pcd.colors = current_pcd.colors
                vis.add_geometry(pcd)
                app_state.previous_pcd = current_pcd
                is_first_frame = False
            else:
                pcd.points = current_pcd.points
                pcd.colors = current_pcd.colors

            if app_state.state == "DETECTING_PLANE":
                # 使用RANSAC算法检测平面
                plane_model, inliers = pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)

                if len(inliers) > 5000:  # 确保平面足够大
                    app_state.plane_model = plane_model
                    # 创建一个绿色的半透明网格来可视化平面
                    plane_pcd = pcd.select_by_index(inliers)
                    app_state.plane_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(plane_pcd,
                                                                                                         0.02)
                    app_state.plane_mesh.paint_uniform_color([0, 1, 0])
                    app_state.plane_mesh.compute_vertex_normals()

                    vis.add_geometry(app_state.plane_mesh, reset_bounding_box=False)
                    print("平面已检测! 按 'P' 键在平面中心放置物体。")
                    app_state.state = "AWAITING_PLACEMENT"
                else:
                    print("未检测到足够大的平面，请调整相机视角。")
                    app_state.state = "IDLE"  # 返回空闲状态

            elif app_state.state == "AWAITING_PLACEMENT":
                # 等待用户按'P'键的下一次回调
                if not vis.get_key_callback(ord("P")):  # 这是一个hack，实际应在回调中处理
                    # 放置物体
                    plane_center = app_state.plane_mesh.get_center()
                    print(f"物体将放置在: {plane_center}")

                    app_state.object_transform[0:3, 3] = plane_center
                    app_state.virtual_object.transform(app_state.object_transform)

                    app_state.state = "TRACKING"
                    print("状态切换 -> TRACKING. 您现在可以移动相机了。")

                    vis.remove_geometry(app_state.plane_mesh, reset_bounding_box=False)
                    vis.add_geometry(app_state.virtual_object, reset_bounding_box=False)
                    app_state.previous_pcd = current_pcd

            elif app_state.state == "TRACKING":
                # 使用ICP进行简单的帧间追踪
                trans_init = np.identity(4)
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    app_state.previous_pcd, current_pcd, 0.02, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())

                # 更新相机位姿
                app_state.camera_pose = np.dot(reg_p2p.transformation, app_state.camera_pose)
                app_state.previous_pcd = current_pcd

                # 更新场景中的所有物体（这里只有点云和虚拟物体）
                pcd.transform(app_state.camera_pose)
                app_state.virtual_object.transform(app_state.camera_pose)

            # 更新可视化
            vis.update_geometry(pcd)
            if app_state.virtual_object:
                vis.update_geometry(app_state.virtual_object)

            if not vis.poll_events():
                break
            vis.update_renderer()

    finally:
        pipeline.stop()
        vis.destroy_window()


if __name__ == "__main__":
    main()