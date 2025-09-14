import open3d as o3d
import numpy as np
import cv2
import mediapipe as mp
from ObTypes import *
import Pipeline
import Context
from Error import ObException
import time

# --- 物理引擎参数 ---
GRAVITY = np.array([0, -9.8, 0])  # Y轴为上方向
TIME_STEP = 1.0 / 30.0  # 假设30fps


class PhysicsObject:
    """一个简单的物理对象模拟器"""

    def __init__(self, mesh, pose=np.identity(4)):
        self.mesh = mesh
        self.pose = pose
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.is_grabbed = False

    def update(self, plane_model):
        if self.is_grabbed:
            return

        # 施加重力
        self.velocity += GRAVITY * TIME_STEP

        # 更新位置
        position = self.pose[:3, 3]
        position += self.velocity * TIME_STEP

        # 碰撞检测 (与平面)
        if plane_model is not None:
            plane_normal = plane_model[:3]
            plane_d = plane_model[3]

            # 获取物体最低点 (这是一个近似值)
            min_bound = self.mesh.get_min_bound()
            lowest_point = self.pose @ np.array([min_bound[0], min_bound[1], min_bound[2], 1])

            distance_to_plane = np.dot(plane_normal, lowest_point[:3]) + plane_d

            if distance_to_plane < 0:
                # 穿透了平面，进行校正
                position -= plane_normal * distance_to_plane
                # 简单的反弹效果
                self.velocity = -0.3 * self.velocity  # 能量损失

        self.pose[:3, 3] = position

    def get_transformed_mesh(self):
        transformed = o3d.geometry.TriangleMesh(self.mesh)
        return transformed.transform(self.pose)


class InteractiveARSystem:
    def __init__(self, intrinsics):
        self.intrinsics = intrinsics
        self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            intrinsics['width'], intrinsics['height'],
            intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy'])

        self.state = "SCANNING"
        self.plane_model = None
        self.plane_mesh_vis = None
        self.physics_object = None

        # --- MediaPipe 手势识别 ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
        self.hand_landmarks = None
        self.is_pinching = False
        self.pinch_pos_3d = None

        # --- 渲染与交互 ---
        self.cv2_window_name = "Interactive AR"
        cv2.namedWindow(self.cv2_window_name)
        cv2.setMouseCallback(self.cv2_window_name, self.mouse_callback)

        # --- 虚拟对象 ---
        try:
            dragon_mesh = o3d.io.read_triangle_mesh("./RGBPoints/Fantasy Dragon.ply")
            dragon_mesh.compute_vertex_normals()
            dragon_mesh.paint_uniform_color([0.8, 0.2, 0.2])
            dragon_mesh.scale(0.3, center=dragon_mesh.get_center())  # 缩小模型
            self.virtual_object_template = dragon_mesh
        except Exception as e:
            print("错误: 无法加载模型 'Fantasy Dragon.ply'。")
            raise e

    def process_frame(self, color_frame, depth_frame):
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_scale = depth_frame.getValueScale()

        self.detect_hand(color_image, depth_image, depth_scale)

        if self.state == "SCANNING":
            self.detect_plane(color_image, depth_image, depth_scale)

        if self.state == "INTERACTING":
            self.update_interaction()
            if self.physics_object:
                self.physics_object.update(self.plane_model)

        output_image = self.render(color_image)
        self.draw_ui(output_image)
        cv2.imshow(self.cv2_window_name, output_image)

    def detect_hand(self, color_image, depth_image, depth_scale):
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        self.hand_landmarks = None
        self.is_pinching = False

        if results.multi_hand_landmarks:
            self.hand_landmarks = results.multi_hand_landmarks[0]

            # 计算食指和拇指指尖的距离
            thumb_tip = self.hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = self.hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

            dist_2d = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

            if dist_2d < 0.05:  # 捏合阈值
                self.is_pinching = True

                # 计算捏合点的3D位置
                pinch_x = int((thumb_tip.x + index_tip.x) / 2 * self.intrinsics['width'])
                pinch_y = int((thumb_tip.y + index_tip.y) / 2 * self.intrinsics['height'])

                depth = depth_image[pinch_y, pinch_x]
                if depth > 0:
                    z = depth / depth_scale
                    x = (pinch_x - self.intrinsics['cx']) * z / self.intrinsics['fx']
                    y = (pinch_y - self.intrinsics['cy']) * z / self.intrinsics['fy']
                    self.pinch_pos_3d = np.array([x, y, z])

    def update_interaction(self):
        if not self.physics_object: return

        if self.is_pinching and self.pinch_pos_3d is not None:
            obj_pos = self.physics_object.pose[:3, 3]
            dist_to_obj = np.linalg.norm(self.pinch_pos_3d - obj_pos)

            if dist_to_obj < 0.1 or self.physics_object.is_grabbed:  # 抓取距离阈值
                self.physics_object.is_grabbed = True
                self.physics_object.pose[:3, 3] = self.pinch_pos_3d
                self.physics_object.velocity = np.array([0.0, 0.0, 0.0])  # 抓住时速度为0
        else:
            self.physics_object.is_grabbed = False

    def detect_plane(self, color_image, depth_image, depth_scale):
        # (与上一个版本类似，但仅用于物理碰撞)
        rgbd = self.create_rgbd_image(color_image, depth_image, depth_scale)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.o3d_intrinsics)
        if not pcd.has_points(): return

        plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
        if len(inliers) > 50000:
            self.plane_model = plane_model
            plane_cloud = pcd.select_by_index(inliers)
            hull, _ = plane_cloud.compute_convex_hull()
            self.plane_mesh_vis = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
            self.plane_mesh_vis.paint_uniform_color([0, 1, 0])

    def render(self, color_image):
        """使用Open3D的离屏渲染器来渲染场景"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.intrinsics['width'], height=self.intrinsics['height'], visible=False)

        # 添加虚拟物体
        if self.state == "INTERACTING" and self.physics_object:
            vis.add_geometry(self.physics_object.get_transformed_mesh())

        # 添加平面预览
        if self.state == "SCANNING" and self.plane_mesh_vis:
            vis.add_geometry(self.plane_mesh_vis)

        ctr = vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        cam_params.intrinsic = self.o3d_intrinsics
        # 保持相机静止
        cam_params.extrinsic = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

        vis.poll_events()
        vis.update_renderer()
        rendered_image = (np.asarray(vis.capture_screen_float_buffer(False)) * 255).astype(np.uint8)
        vis.destroy_window()

        rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
        mask = np.sum(rendered_image, axis=2) > 0

        final_image = color_image.copy()
        final_image[mask] = rendered_image[mask]
        return final_image

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.state == "SCANNING" and self.plane_model is not None:
            # (与上个版本类似，用于放置第一个物体)
            cam_x = (x - self.intrinsics['cx']) / self.intrinsics['fx']
            cam_y = (y - self.intrinsics['cy']) / self.intrinsics['fy']
            ray_dir = np.array([cam_x, cam_y, 1.0]) / np.linalg.norm(np.array([cam_x, cam_y, 1.0]))

            denom = np.dot(self.plane_model[:3], ray_dir)
            if abs(denom) > 1e-6:
                t = -self.plane_model[3] / denom
                point = t * ray_dir

                pose = np.identity(4)
                pose[:3, 3] = point + np.array([0, 0.1, 0])  # 稍微抬高一点，让它自然落下
                self.physics_object = PhysicsObject(self.virtual_object_template, pose)

                self.state = "INTERACTING"

    def draw_ui(self, image):
        # 绘制手部关键点
        if self.hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(image, self.hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        font, color = cv2.FONT_HERSHEY_SIMPLEX, (0, 255, 0)
        if self.state == "SCANNING":
            if self.plane_model is None:
                cv2.putText(image, "扫描平面中...", (20, 40), font, 1, color, 2)
            else:
                cv2.putText(image, "平面已识别! 点击屏幕放置物体", (20, 40), font, 1, color, 2)
        elif self.state == "INTERACTING":
            text = "交互模式: "
            if self.physics_object and self.physics_object.is_grabbed:
                text += "已抓取"
            else:
                text += "自由模式"
            cv2.putText(image, text, (20, 40), font, 1, color, 2)
            cv2.putText(image, "按 'R' 重置", (20, 80), font, 1, color, 2)

    def reset(self):
        self.state = "SCANNING"
        self.plane_model = None
        self.physics_object = None

    def create_rgbd_image(self, color_image, depth_image, depth_scale):
        # (与上个版本相同)
        o3d_color = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        o3d_depth = o3d.geometry.Image(depth_image)
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1.0 / depth_scale, depth_trunc=4.0, convert_rgb_to_intensity=False)


if __name__ == "__main__":
    try:
        pipeline = Pipeline.Pipeline(None, None)
        config = Pipeline.Config()
        color_profile = pipeline.getStreamProfileList(OB_PY_SENSOR_COLOR).getVideoStreamProfile(640, 480,
                                                                                                OB_PY_FORMAT_BGR, 30)
        depth_profile = pipeline.getStreamProfileList(OB_PY_SENSOR_DEPTH).getProfile(0)
        config.enableStream(color_profile)
        config.enableStream(depth_profile)
        config.setAlignMode(OB_PY_ALIGN_D2C_HW_MODE)
        pipeline.start(config, None)

        cam_param = pipeline.getCameraParam()
        intrinsics = {
            'width': color_profile.width(), 'height': color_profile.height(),
            'fx': cam_param.rgbIntrinsic.fx, 'fy': cam_param.rgbIntrinsic.fy,
            'cx': cam_param.rgbIntrinsic.cx, 'cy': cam_param.rgbIntrinsic.cy
        }

        ar_system = InteractiveARSystem(intrinsics)

        while True:
            frameset = pipeline.waitForFrames(100)
            if frameset is None: continue
            color_frame, depth_frame = frameset.colorFrame(), frameset.depthFrame()
            if color_frame is None or depth_frame is None: continue

            ar_system.process_frame(color_frame, depth_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('r'): ar_system.reset()

    finally:
        if 'pipeline' in locals(): pipeline.stop()
        cv2.destroyAllWindows()