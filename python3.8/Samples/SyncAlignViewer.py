from ObTypes import *
from Property import *
import Pipeline
import StreamProfile
import Device
from Error import ObException
import cv2
import numpy as np
import sys
import math

q = 113
ESC = 27

D = 68
d = 100
F = 70
f = 102
S = 83
s = 115
add = 43
reduce = 45
FEMTO = 0x0635

sync	= False
started = True
hd2c	= False
sd2c	= True
alpha   = 0.5
keyRecord  = -1

frameSet   = None
colorFrame = None
depthFrame = None

#  Synchronous alignment example
# 
# This example may be due to the depth or color sensor does not support mirroring and
# the depth image and color image mirroring state is not the same, resulting in the depth 
# image and color image display image is opposite, if encountered, then by setting the mirroring
# interface to keep the two mirroring state can be consistent In addition, there may be 
# some devices to obtain the resolution does not support the D2C function, so the D2C function
# is based on the actual support D2C resolution Therefore, the D2C function is based on the 
# actual supported D2C resolution.

# For instance, while DaBai DCW or GeminiE supports a resolution of 640x360 for D2C, the actual
# resolution setting in a given example may be 640x480. Therefore, the user can adjust the 
# resolution based on the specific module to make the corresponding 640x360 resolution work properly.

try:
  # Create a Pipeline, which is the entry point for the entire advanced API and can be easily opened 
  # and closed through the Pipeline multiple types of streams and get a set of frame data
  pipe = Pipeline.Pipeline(None, None)
  # Configure which streams to enable or disable in Pipeline by creating a Config
  config = Pipeline.Config()

  try:
    # Get all stream configurations for the color camera, including the stream resolution, 
    # frame rate, and frame format
    profiles = pipe.getStreamProfileList(OB_PY_SENSOR_COLOR)

    videoProfile = None
    try:
      # Select the default resolution to open the stream, 
      # you can configure the default resolution through the configuration file
      videoProfile = profiles.getProfile(0)
    except ObException as e:
      print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
      
    colorProfile = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
    config.enableStream(colorProfile)
  except ObException as e:
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
    print("Current device is not support color sensor!")
    sys.exit()

  try:
    # Get all stream configurations for the depth camera, including the stream resolution, 
    # frame rate, and frame format
    profiles = pipe.getStreamProfileList(OB_PY_SENSOR_DEPTH)

    videoProfile = None
    try:
      # Select the default resolution to open the stream, you can configure the default resolution
      # through the configuration file
      videoProfile = profiles.getProfile(0)
    except ObException as e:
      print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
      
    depthProfile = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
    config.enableStream(depthProfile)
  except ObException as e:
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
    print("Current device is not support depth sensor!")
    sys.exit()

  # Configure alignment mode to software D2C alignment
  config.setAlignMode(OB_PY_ALIGN_D2C_SW_MODE)

  try:
    # Start the stream configured in Config, if no parameters are passed,
    # it will start the default configuration start stream
    pipe.start(config, None)
  except ObException as e:
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))

  while True:
    frameSet = None
    colorFrame = None
    depthFrame = None
    key = cv2.waitKey(1)
    # Press + to increase alpha
    if keyRecord != key and key == add :
      alpha += 0.01
      if alpha >= 1.0 :
        alpha = 1.0

    # Press the - key to reduce alpha
    if keyRecord != key and key == reduce :
      alpha -= 0.01
      if alpha <= 0.0 :
        alpha = 0.0
    
    # Press the D key to switch hardware D2C
    if keyRecord != key and (key == D or key == d) :
      try:
        if hd2c == False:
          started = False
          pipe.stop()
          hd2c = True
          sd2c = False
          config.setAlignMode(OB_PY_ALIGN_D2C_HW_MODE)
          pipe.start(config, None)
          started = True
        else:
          started = False
          pipe.stop()
          hd2c = False
          sd2c = False
          config.setAlignMode(OB_PY_ALIGN_DISABLE)
          pipe.start(config, None)
          started = True
      except ObException as e:
        print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
        print("Property not support")

    # Press S key to switch software D2C
    if keyRecord != key and (key == S or key == s) :
      try:
        if sd2c == False:
          started = False
          pipe.stop()
          sd2c= True
          hd2c = False
          config.setAlignMode(OB_PY_ALIGN_D2C_SW_MODE)
          pipe.start(config, None)
          started = True
        else:
          started = False
          pipe.stop()
          hd2c = False
          sd2c = False
          config.setAlignMode(OB_PY_ALIGN_DISABLE)
          pipe.start(config, None)
          started = True
      except ObException as e:
        print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
        print("Property not support")

    # Press F key to switch synchronization
    if keyRecord != key and (key == F or key == f) :
      sync = bool(1 - sync)
      if sync :
        try:
          pipe.enableFrameSync()
        except ObException as e:
          print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
          print("Sync not support")					
      else :
        try:
          pipe.disableFrameSync()
        except ObException as e:
          print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
          print("Sync not support")	

    keyRecord = key

    # 以阻塞的方式等待一帧数据，该帧是一个复合帧，里面包含配置里启用的所有流的帧数据，
    # 并设置帧的等待超时时间为100ms
    frameSet = pipe.waitForFrames(100)

    if frameSet == None:
      continue
    else:
      # 在窗口中渲染一组帧数据，这里将渲染彩色帧及深度帧，将彩色帧及深度帧叠加显示
      colorFrame = frameSet.colorFrame()
      depthFrame = frameSet.depthFrame()

      if colorFrame != None and depthFrame != None:
        # 获取帧的大小、数据、宽高
        colorSize = colorFrame.dataSize()
        colorData = colorFrame.data()
        depthSize = depthFrame.dataSize()
        depthData = depthFrame.data()
        colorWidth = colorFrame.width()
        colorHeight = colorFrame.height()
        colorFormat = colorFrame.format()
        depthWidth = depthFrame.width()
        depthHeight = depthFrame.height()
        valueScale = depthFrame.getValueScale()
    
        if colorData is not None and depthData is not None:
          newColorData = colorData
          # Resize the color frame data to (height,width,3)
          if colorFormat == OB_PY_FORMAT_MJPG:
              # 将数据帧MJPG解码为RGB格式
              newColorData = cv2.imdecode(newColorData,1)
              # Decode data frame MJPG to RGB format
              if newColorData is not None:
                newColorData = np.resize(newColorData,(colorHeight, colorWidth, 3))
          elif colorFormat == OB_PY_FORMAT_RGB888:
              # Decode data frame MJPG to RGB format
              newColorData = np.resize(newColorData,(colorHeight, colorWidth, 3))
              # Convert frame data RGB to BGR
              newColorData = cv2.cvtColor(newColorData, cv2.COLOR_RGB2BGR)
          elif colorFormat == OB_PY_FORMAT_YUYV:
              newColorData = np.resize(newColorData,(colorHeight, colorWidth, 2))
              # Converting data frame YUYV to RGB data
              newColorData = cv2.cvtColor(newColorData, cv2.COLOR_YUV2BGR_YUYV)
          elif colorFormat == OB_PY_FORMAT_UYVY:
              newColorData = np.resize(newColorData,(colorHeight, colorWidth, 2))
              # Converting data frame YUYV to RGB data
              newColorData = cv2.cvtColor(newColorData, cv2.COLOR_YUV2BGR_UYVY)
          elif colorFormat == OB_PY_FORMAT_I420:
              newColorData = newColorData.reshape((colorHeight * 3 // 2, colorWidth))
              newColorData = cv2.cvtColor(newColorData, cv2.COLOR_YUV2BGR_I420)
              newColorData = cv2.resize(newColorData, (colorWidth, colorHeight))

          # Resize the depth frame data to (height,width,2)
          depthData = np.resize(depthData,(depthHeight, depthWidth, 2))
          
          # Convert frame data from 8bit to 16bit
          newDepthData = depthData[:,:,0]+depthData[:,:,1]*256          
          # Convert frame data to 1mm units
          newDepthData = (newDepthData * valueScale).astype('uint16')
          normalized_image = (newDepthData / 32).astype('uint8')
          
          # Convert depth frame data GRAY to RGB
          outputDepthImage = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB) 
          
          # Need to scale to the same resolution when the depth is not the same as the resolution of the color
          if colorHeight != depthHeight:
            outputDepthImage = cv2.resize(outputDepthImage,(colorWidth,colorHeight))
            
            
          if newColorData is not None:
            newData = newColorData
          if outputDepthImage is not None:
            newData = outputDepthImage
          if newColorData is not None and outputDepthImage is not None:
            newData = cv2.addWeighted(newColorData, (1 - alpha), outputDepthImage, alpha, 0)
          
          cv2.namedWindow("SyncAlignViewer", cv2.WINDOW_NORMAL)

          cv2.imshow("SyncAlignViewer", newData)

          if key == ESC or key == q:
            cv2.destroyAllWindows()
            break

  pipe.stop()

except ObException as e:
  print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))