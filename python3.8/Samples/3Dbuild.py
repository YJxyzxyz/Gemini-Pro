import numpy as np
import cv2
import open3d as o3d
import os

def generate_point_cloud(rgb_image_path, depth_image_path, fx, fy, cx, cy, depth_scale=1.0):
    rgb_image = cv2.imread(rgb_image_path)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    if rgb_image is None or depth_image is None:
        raise ValueError("无法加载图像，请检查路径和文件。")

    # 如果深度图为多通道，转换为单通道灰度图
    if len(depth_image.shape) == 3:
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)

    h, w = depth_image.shape
    assert rgb_image.shape[:2] == (h, w), "深度图和 RGB 图像尺寸不一致！"

    # 转换深度图格式为 float32
    depth_image = depth_image.astype(np.float32)

    # 平滑深度图
    depth_image = cv2.bilateralFilter(depth_image, d=5, sigmaColor=75, sigmaSpace=75)

    # 转换深度单位
    depth_image /= depth_scale

    # 构建网格坐标
    i, j = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    valid_mask = depth_image > 0
    x = (j[valid_mask] - cx) * depth_image[valid_mask] / fx
    y = (i[valid_mask] - cy) * depth_image[valid_mask] / fy
    z = depth_image[valid_mask]

    # 生成 3D 点
    points = np.stack((x, y, z), axis=-1)
    colors = rgb_image[valid_mask] / 255.0

    # 构造点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # 点云降噪
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    filtered_point_cloud = point_cloud.select_by_index(ind)
    return filtered_point_cloud


def main():
    rgb_image_path = "./saved_images/color_image_0.png"
    depth_image_path = "./saved_images/depth_image_0.png"

    fx, fy = 525.0, 525.0
    cx, cy = 319.5, 239.5
    depth_scale = 1000.0

    point_cloud = generate_point_cloud(rgb_image_path, depth_image_path, fx, fy, cx, cy, depth_scale)

    # 显示点云
    o3d.visualization.draw_geometries([point_cloud])

    # 创建保存点云的文件夹
    output_folder = "./pointcloud"
    os.makedirs(output_folder, exist_ok=True)

    # 保存点云
    output_path = os.path.join(output_folder, "output_point_cloud_filtered.ply")
    o3d.io.write_point_cloud(output_path, point_cloud)
    print(f"过滤后的点云已保存至 {output_path}")


if __name__ == "__main__":
    main()
