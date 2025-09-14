# 💎 Gemini Pro 视觉工具箱 (Gemini Pro Vision Toolkit)

这是一个基于奥比中光 **Gemini Pro** 结构光相机的多功能计算机视觉开发套件。本项目从基础的图像与点云数据采集出发，逐步扩展到高级的三维重建、AI物体识别，最终实现了一个具有物理模拟和手势控制的增强现实（AR）互动沙盒。

## ✨ 功能概览

| 功能模块                | 脚本文件                                           | 描述                                                 |
| ----------------------- | -------------------------------------------------- | ---------------------------------------------------- |
| **📸 基础相机操作**      | `HelloOrbbec.py`, `SaveDepth.py`, `MultiDevice.py` | 获取设备信息，实时显示RGB/深度流，并保存图像。       |
| **☁️ 点云生成与显示**    | `PointCloud.py`, `PointCloudShow.py`               | 从深度数据生成彩色/非彩色点云，并进行可视化。        |
| **🏗️ 静态三维重建**      | `3Dbuild.py`                                       | 使用单帧RGB和深度图重建场景的三维点云模型。          |
| **🚀 实时三维重建**      | `Live3DReconstruction.py`                          | 实时拼接多帧点云，动态构建更大范围的三维场景。       |
| **📏 交互式三维测量**    | `3DMeasurementTool.py`                             | 在三维点云中交互式地测量任意两点间的空间距离。       |
| **✋ AI手势识别**        | `GestureRecognition.py`                            | 利用MediaPipe实时识别手指数和“点赞”等手势。          |
| **🤖 AI物体识别与分割**  | `ObjectSegmentation3D.py`                          | 结合YOLOv5模型，在场景中识别物体并分割出其3D点云。   |
| **🕶️ 高级增强现实 (AR)** | `AdvancedAR.py`                                    | 具备真实感遮挡和精确放置能力的AR系统。               |
| **💥 终极AR互动沙盒**    | `InteractiveAR.py`                                 | 融合物理引擎与手势控制，实现对虚拟物体的抓取和抛掷。 |

## 🛠️ 环境配置

在运行任何脚本之前，请确保您已配置好Python环境并安装所有必要的库。

1. **基础依赖**:

   Bash

   ```
   pip install numpy opencv-python matplotlib open3d
   ```

2. **AI与高级功能依赖**:

   - **PyTorch**: 请访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 根据您的系统（Windows/Linux）和硬件（CUDA/CPU）选择并运行对应的安装命令。
   - **MediaPipe & OpenCV Contrib**:

   Bash

   ```
   pip install mediapipe opencv-contrib-python
   ```

3. **硬件与SDK**:

   - 奥比中光 Gemini Pro 相机
   - 官方提供的Orbbec SDK（本项目已包含Python封装好的库文件）

## 🚀 程序运行指南

所有脚本都位于 `Samples` 目录下。

### 1. 基础功能

#### `HelloOrbbec.py`

获取并打印已连接的Gemini Pro相机的详细信息。

Bash

```
python HelloOrbbec.py
```

#### `SaveDepth.py` & `MultiDevice.py`

实时显示深度或RGB+深度视频流。在窗口激活时按下 `S` 键可将当前帧保存到 `depth_images` 或 `saved_images` 文件夹。

Bash

```
python SaveDepth.py
# 或
python MultiDevice.py
```

### 2. 点云与三维重建

#### `PointCloud.py`

在运行时，按下 `R` 键生成并保存带颜色的点云文件 (`RGBPoints.ply`)，按下 `D` 键保存深度点云 (`DepthPoints.ply`)。

Bash

```
python PointCloud.py
```

#### `3Dbuild.py`

利用 `saved_images` 文件夹中的图像进行一次性的三维重建，并显示结果。

Bash

```
python 3Dbuild.py
```

#### `Live3DReconstruction.py`

打开一个实时三维建图窗口。您可以缓慢移动相机来扫描并构建一个更大的场景。

Bash

```
python Live3DReconstruction.py
```

### 3. AI与高级交互应用

#### `3DMeasurementTool.py`

程序会先捕捉一帧场景并显示点云。在3D窗口中，按住 `SHIFT` 并用鼠标左键点选两点，即可在终端看到它们之间的距离。

Bash

```
python 3DMeasurementTool.py
```

#### `GestureRecognition.py`

实时识别视频流中的手势。程序能够识别伸出的手指数和“竖起大拇指”的手势，并将结果显示在屏幕上。

Bash

```
python GestureRecognition.py
```

#### `ObjectSegmentation3D.py`

结合YOLOv5模型，程序会打开两个窗口：一个实时显示2D物体检测结果，另一个则显示从场景中分割出的物体的3D点云。

Bash

```
python ObjectSegmentation3D.py
```

### 4. 增强现实 (AR) 系列

#### `AdvancedAR.py`

一个功能完善的AR应用。

1. 运行后将相机对准一个平面（如桌面）。
2. 程序识别平面后，用鼠标左键点击屏幕以**精确放置**虚拟模型。
3. 放置后，您可以移动相机，虚拟模型会保持在原位。
4. 将您的手或其他真实物体放在模型前，可以观察到**真实的遮挡效果**。
5. 按 `R` 键重置，`Q` 键退出。

Bash

```
python AdvancedAR.py
```

#### `InteractiveAR.py` (🌟 **最终版** 🌟)

这是一个融合了物理引擎和手势控制的终极AR体验。

1. 启动后，首先扫描并点击屏幕放置物体。物体会受重力下落到平面上。
2. 将您的手伸到摄像头前，做出**捏合**手势（食指与拇指指尖靠近）。
3. 用捏合的手势靠近虚拟物体即可**抓取**它。
4. 移动您的手来移动物体，松开手指即可**释放**或**抛出**物体。
5. 按 `R` 键重置，`Q` 键退出。

Bash

```
python InteractiveAR.py
```

## 🤝 贡献与致谢

- 感谢 [奥比中光(Orbbec)](https://vcp.developer.orbbec.com.cn/documentation) 提供的优秀相机和SDK。
- 本项目中使用的AI模型和算法库归其原作者所有（YOLOv5, MediaPipe, Open3D等）。
