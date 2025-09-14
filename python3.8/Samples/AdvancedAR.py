import open3d as o3d
import numpy as np
import cv2
from ObTypes import *
import Pipeline
import Context
from Error import ObException


class AdvancedARSystem:
    def __init__(self, intrinsics):
        self.intrinsics = intrinsics
        self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            intrinsics['width'], intrinsics['height'],
            intrinsics['fx'], intrinsics['fy'],
            intrinsics['cx'], intrinsics['cy']
        )

        # --- 状态管理 ---
        self.state = "SCANNING"
        self.plane_model = None
        self.plane_mesh = None

        # --- 虚拟对象 ---
        try:
            self.virtual_object_original = o3d.io.read_triangle_mesh("./RGBPoints/Fantasy Dragon.ply")
            self.virtual_object_original.compute_vertex_normals()
            self.virtual_object_original.paint_uniform_color([0.8, 0.2, 0.2])
        except Exception as e:
            print(f"错误: 无法加载模型。请确保 'Fantasy Dragon.ply' 存在于 'RGBPoints' 文件夹中。")
            raise e
        self.virtual_object = None  # 场景中的实例

        # --- 视觉追踪 ---
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.reference_frame = None
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.camera_pose = np.identity(4)
        self.object_world_pose = np.identity(4)

        # --- 渲染与交互 ---
        self.cv2_window_name = "AR View"
        cv2.namedWindow(self.cv2_window_name)
        cv2.setMouseCallback(self.cv2_window_name, self.mouse_callback)

    def process_frame(self, color_frame, depth_frame):
        """处理每一帧数据"""
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_scale = depth_frame.getValueScale()

        if self.state == "SCANNING":
            self.detect_plane(color_image, depth_image, depth_scale)
        elif self.state == "TRACKING":
            self.track_camera(color_image, depth_image, depth_scale)

        # 最终渲染
        output_image = self.render(color_image, depth_image, depth_scale)
        self.draw_ui(output_image)
        cv2.imshow(self.cv2_window_name, output_image)

    def detect_plane(self, color_image, depth_image, depth_scale):
        """使用RANSAC检测主平面"""
        rgbd = self.create_rgbd_image(color_image, depth_image, depth_scale)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.o3d_intrinsics)

        if not pcd.has_points():
            return

        plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)

        if len(inliers) > pcd.get_max_bound()[0] * pcd.get_max_bound()[1] * 5000:  # 确保平面够大
            self.plane_model = plane_model
            plane_cloud = pcd.select_by_index(inliers)
            plane_cloud.paint_uniform_color([0, 1, 0])  # 绿色

            # 使用凸包来可视化平面范围
            try:
                hull, _ = plane_cloud.compute_convex_hull()
                hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
                hull_ls.paint_uniform_color([0, 1, 0])
                self.plane_mesh = hull_ls  # 用线框显示
            except:
                self.plane_mesh = plane_cloud

    def track_camera(self, color_image, depth_image, depth_scale):
        """基于ORB特征追踪相机位姿"""
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray_image, None)

        if des is None: return

        matches = self.bf_matcher.match(self.reference_descriptors, des)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 30:  # 至少需要30个匹配点
            # 获取匹配点的3D坐标
            ref_pts_3d = []
            cur_pts_2d = []

            ref_depth_map = self.reference_frame['depth']

            for m in matches:
                u, v = self.reference_keypoints[m.queryIdx].pt
                d = ref_depth_map[int(v), int(u)]

                if d > 0:
                    z = d / depth_scale
                    x = (u - self.intrinsics['cx']) * z / self.intrinsics['fx']
                    y = (v - self.intrinsics['cy']) * z / self.intrinsics['fy']
                    ref_pts_3d.append([x, y, z])
                    cur_pts_2d.append(kp[m.trainIdx].pt)

            if len(ref_pts_3d) > 15:  # 至少需要15个有效3D点
                ref_pts_3d = np.array(ref_pts_3d, dtype=np.float32)
                cur_pts_2d = np.array(cur_pts_2d, dtype=np.float32)

                # 使用PnP算法求解相机位姿变换
                _, rvec, tvec, _ = cv2.solvePnPRansac(ref_pts_3d, cur_pts_2d,
                                                      np.array(self.intrinsics['mtx']),
                                                      np.array(self.intrinsics['dist']))

                if rvec is not None and tvec is not None:
                    R, _ = cv2.Rodrigues(rvec)
                    transform = np.identity(4)
                    transform[:3, :3] = R
                    transform[:3, 3] = tvec.flatten()
                    self.camera_pose = np.linalg.inv(transform)

    def render(self, color_image, depth_image, depth_scale):
        """渲染虚拟物体并处理遮挡"""
        if self.state != "TRACKING" or self.virtual_object is None:
            return color_image

        # --- 核心遮挡逻辑 ---
        # 1. 创建一个自定义的Open3D渲染器
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.intrinsics['width'], height=self.intrinsics['height'], visible=False)
        vis.add_geometry(self.virtual_object)

        # 2. 设置渲染器的相机视角以匹配真实相机
        ctr = vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()

        # 将追踪到的相机位姿应用到渲染器相机
        extrinsic = np.linalg.inv(self.object_world_pose) @ np.linalg.inv(self.camera_pose)
        cam_params.extrinsic = extrinsic
        cam_params.intrinsic = self.o3d_intrinsics
        ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

        # 3. 渲染虚拟物体的彩色图和深度图
        vis.poll_events()
        vis.update_renderer()
        virtual_color = (np.asarray(vis.capture_screen_float_buffer(False)) * 255).astype(np.uint8)
        virtual_depth_o3d = vis.capture_depth_float_buffer(False)
        vis.destroy_window()

        virtual_color = cv2.cvtColor(virtual_color, cv2.COLOR_RGB2BGR)
        virtual_depth = (np.asarray(virtual_depth_o3d) * depth_scale).astype(np.uint16)

        # 4. 混合渲染
        # 创建一个掩码，标记虚拟物体出现的位置 (深度>0)
        mask = virtual_depth > 0

        # 获取真实深度
        real_depth = depth_image

        # 遮挡掩码：真实物体比虚拟物体更近的地方
        occlusion_mask = (real_depth > 0) & (real_depth < virtual_depth)

        # 从虚拟物体掩码中移除被遮挡的部分
        mask[occlusion_mask] = False

        # 将虚拟物体绘制到真实图像上
        final_image = color_image.copy()
        final_image[mask] = virtual_color[mask]

        return final_image

    def mouse_callback(self, event, x, y, flags, param):
        """处理鼠标点击事件"""
        if event == cv2.EVENT_LBUTTONDOWN and self.state == "SCANNING" and self.plane_model is not None:
            print("收到放置请求...")

            # --- 精确放置：Ray-Casting ---
            # 1. 将2D屏幕坐标转换为相机坐标系下的归一化坐标
            cam_x = (x - self.intrinsics['cx']) / self.intrinsics['fx']
            cam_y = (y - self.intrinsics['cy']) / self.intrinsics['fy']

            # 2. 创建射线方向向量
            ray_direction = np.array([cam_x, cam_y, 1.0])
            ray_direction = ray_direction / np.linalg.norm(ray_direction)

            # 3. 计算射线与平面的交点 (数学公式)
            # Plane: ax + by + cz + d = 0, Ray: P = t*D
            # t = -d / (a*Dx + b*Dy + c*Dz)
            plane_normal = self.plane_model[:3]
            plane_d = self.plane_model[3]

            # 射线原点是相机中心 (0,0,0)
            denominator = np.dot(plane_normal, ray_direction)

            if abs(denominator) > 1e-6:  # 避免除以零
                t = -plane_d / denominator
                intersection_point = t * ray_direction

                print(f"物体成功放置于3D坐标: {intersection_point}")

                # 记录物体世界坐标并实例化
                self.object_world_pose[:3, 3] = intersection_point
                self.virtual_object = o3d.geometry.TriangleMesh(self.virtual_object_original)  # 创建一个副本
                self.virtual_object.transform(self.object_world_pose)

                # 设置参考帧并切换到追踪状态
                frameset = pipeline.waitForFrames(100)
                self.reference_frame = {
                    'color': np.asanyarray(frameset.colorFrame().get_data()),
                    'depth': np.asanyarray(frameset.depthFrame().get_data())
                }
                gray_ref = cv2.cvtColor(self.reference_frame['color'], cv2.COLOR_BGR2GRAY)
                self.reference_keypoints, self.reference_descriptors = self.orb.detectAndCompute(gray_ref, None)
                self.state = "TRACKING"

    def draw_ui(self, image):
        """在屏幕上绘制UI文本"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0)

        if self.state == "SCANNING":
            if self.plane_model is None:
                cv2.putText(image, "扫描中... 请将相机对准一个平坦表面", (20, 40), font, 0.8, color, 2)
            else:
                cv2.putText(image, "平面已识别! 请用鼠标点击以放置物体", (20, 40), font, 0.8, color, 2)
                # 实时渲染平面预览
                if self.plane_mesh is not None:
                    # 这个功能需要更复杂的渲染循环，暂时用文字提示
                    pass
        elif self.state == "TRACKING":
            cv2.putText(image, "追踪模式 | 按 'R' 键重置", (20, 40), font, 0.8, color, 2)

    def reset(self):
        """重置系统状态"""
        self.state = "SCANNING"
        self.plane_model = None
        self.plane_mesh = None
        self.virtual_object = None
        self.camera_pose = np.identity(4)
        print("系统已重置。")

    def create_rgbd_image(self, color_image, depth_image, depth_scale):
        o3d_color = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        o3d_depth = o3d.geometry.Image(depth_image)
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1.0 / depth_scale, depth_trunc=4.0, convert_rgb_to_intensity=False)


if __name__ == "__main__":
    try:
        # --- 初始化相机和SDK ---
        pipeline = Pipeline.Pipeline(None, None)
        config = Pipeline.Config()

        # 尝试获取配置
        color_profile = pipeline.getStreamProfileList(OB_PY_SENSOR_COLOR).getVideoStreamProfile(640, 480,
                                                                                                OB_PY_FORMAT_BGR, 30)
        depth_profile = pipeline.getStreamProfileList(OB_PY_SENSOR_DEPTH).getProfile(0)
        config.enableStream(color_profile)
        config.enableStream(depth_profile)
        config.setAlignMode(OB_PY_ALIGN_D2C_HW_MODE)
        pipeline.start(config, None)

        camera_param = pipeline.getCameraParam()
        intrinsics = {
            'width': color_profile.width(), 'height': color_profile.height(),
            'fx': camera_param.rgbIntrinsic.fx, 'fy': camera_param.rgbIntrinsic.fy,
            'cx': camera_param.rgbIntrinsic.cx, 'cy': camera_param.rgbIntrinsic.cy,
            'mtx': [[camera_param.rgbIntrinsic.fx, 0, camera_param.rgbIntrinsic.cx],
                    [0, camera_param.rgbIntrinsic.fy, camera_param.rgbIntrinsic.cy],
                    [0, 0, 1]],
            'dist': [0, 0, 0, 0, 0]  # 假设无畸变
        }

        # --- 创建并运行AR系统 ---
        ar_system = AdvancedARSystem(intrinsics)

        while True:
            frameset = pipeline.waitForFrames(100)
            if frameset is None: continue

            color_frame = frameset.colorFrame()
            depth_frame = frameset.depthFrame()
            if color_frame is None or depth_frame is None: continue

            ar_system.process_frame(color_frame, depth_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                ar_system.reset()

    except Exception as e:
        print(f"程序发生严重错误: {e}")
    finally:
        if 'pipeline' in locals():
            pipeline.stop()
        cv2.destroyAllWindows()