# Gemini Pro

结构光相机Gemini Pro的一些实用

## 环境配置

python 3.8.5

numpy 1.24.4

opencv-python 4.10.0.84

matplotlib 3.7.5

open3d

## 程序说明

在Samples路径下有奥比官方的例程代码

具体可参考：

[[奥比中光AI开放平台|全球首个聚焦3D视觉开放平台](https://vcp.developer.orbbec.com.cn/documentation)](SaveDepth.py)

除此以外新增了一些功能

SaveDepth.py：实时输出深度图像流，用户按下s时可以保存图像。路径在depth_images

![image](https://github.com/YJxyzxyz/Gemini-Pro/edit/master/python3.8/Samples/depth_images/depth_frame_0.png)

PointCloud.py：修改了代码的部分逻辑，现在ply文件会保存在指定路径

PointCloudShow.py：可视化ply文件

MultiDevice.py：同时输出RGB图流和深度图流，并且按下S键会保存在指定文件夹中

3Dbuild.py：利用深度图和RGB原图完成三维重建并保存点云文件

![image](https://github.com/YJxyzxyz/Gemini-Pro/edit/master/python3.8/Samples/saved_images/color_image.png)
![image](https://github.com/YJxyzxyz/Gemini-Pro/edit/master/python3.8/Samples/saved_images/depth_image.png)
![image](https://github.com/YJxyzxyz/Gemini-Pro/edit/master/python3.8/Samples/saved_images/pointcloud.png)
