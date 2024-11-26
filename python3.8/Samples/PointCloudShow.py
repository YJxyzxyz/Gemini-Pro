import open3d as o3d
import numpy as np

# 加载点云文件
pcd = o3d.io.read_point_cloud("./RGBPoints/Fantasy Dragon.ply")
o3d.visualization.draw_geometries([pcd])
# # 获取点的坐标
# points = np.asarray(pcd.points)
# print("Points data:")
# print(points[:5])  # 打印前5个点
#
# # 获取颜色信息（如果存在）
# if pcd.has_colors():
#     colors = np.asarray(pcd.colors)
#     print("Colors data:")
#     print(colors[:5])  # 打印前5个颜色
#
# # 获取法线信息（如果有）
# if pcd.has_normals():
#     normals = np.asarray(pcd.normals)
#     print("Normals data:")
#     print(normals[:5])  # 打印前5个法线
