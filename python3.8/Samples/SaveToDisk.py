from ObTypes import *
from Property import *
import Context
import Pipeline
import StreamProfile
from Error import ObException
import cv2
import numpy as np

import sys
import platform

plat = platform.system().lower()
if plat == 'windows':
  import msvcrt
elif plat == 'linux':
  import termios, tty

ESC = 27

try:
  ctx = Context.Context(None)
  ctx.setLoggerSeverity(OB_PY_LOG_SEVERITY_ERROR)

  # Create pipeline
  pipe = Pipeline.Pipeline(None, None)
  # Configure which streams to enable or disable in Pipeline by creating a Config
  config = Pipeline.Config()

  colorCount = 0
  depthCount = 0

  try:
    # Get all stream configurations for the color camera, including the stream resolution, frame rate, and frame format
    profiles = pipe.getStreamProfileList(OB_PY_SENSOR_COLOR)

    videoProfile = None
    try:
      # Select the default resolution to open the stream, you can configure the default resolution through the configuration file
      videoProfile = profiles.getProfile(0)
    except ObException as e:
      print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
      
    colorProfile = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
    config.enableStream(colorProfile)
  except ObException as e:
    # No Color Sensor
    colorCount = -1
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
    print("Current device is not support color sensor!")

  try:
    # Get all stream configurations for the depth camera, including the stream resolution, frame rate, and frame format
    profiles = pipe.getStreamProfileList(OB_PY_SENSOR_DEPTH)

    videoProfile = None
    try:
      # Select the default resolution to open the stream,
      # you can configure the default resolution through the configuration file
      videoProfile = profiles.getProfile(0)
    except ObException as e:
      print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
      
    depthProfile = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
    config.enableStream(depthProfile)
  except ObException as e:
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
    print("Current device is not support depth sensor!")
    sys.exit()

  pipe.start(config, None)

  frameCount = 0

  while True:
    # Press ESC to exit the program when 5 color and depth images are saved
    if depthCount == 5 and (colorCount == 5 or colorCount == -1):
      print("The demo is over, please press ESC to exit manually!")
      if plat == 'windows':
        if ord(msvcrt.getch()) == ESC:
          break
      elif plat == 'linux':
        assert sys.stdin.isatty(), "Can't run without a console to run on"
        fd = sys.stdin.fileno()
        stash = termios.tcgetattr(fd)
        try:
          tty.setraw(fd)
          newterm = termios.tcgetattr(fd)
          newterm[tty.LFLAG] |= termios.ECHO
          termios.tcsetattr(fd,termios.TCSADRAIN,newterm)
          message = b''
          while True:
            ch = sys.stdin.buffer.read(1)
            if ch == b'\x1b':
              exit(0)
            elif ch == b'\n':
              break
            else:
              message += ch
          message = message.decode(sys.stdin.encoding)
        finally:
          termios.tcsetattr(fd, termios.TCSANOW, stash)

    # Wait for a frame of data with a timeout of 100ms
    frameset = pipe.waitForFrames(100)
    if frameset == None:
      print("The frameset is null!")
      continue
    # Filter the first 5 frames of data, wait for the data to stabilize and then save
    if frameCount < 5:
      frameCount += 1
      continue

    # Get color and depth frames
    colorFrame = frameset.colorFrame()
    depthFrame = frameset.depthFrame()

    if colorFrame != None and colorCount < 5:
      # Save color images
      colorCount += 1
      colorRawName = "color_" + str(colorFrame.timeStamp()) + ".raw"
      colorPngName = "color_" + str(colorFrame.timeStamp()) + ".png"
      data = colorFrame.data()
      colorWidth = colorFrame.width()
      colorHeight = colorFrame.height()
      colorMat = data
      if colorFrame.format() == OB_PY_FORMAT_MJPG:
          # Decode data frame MJPG to RGB format
          colorMat = cv2.imdecode(colorMat,1)
          # Resize the frame data to (height,width,3)
          if colorMat is not None:
            colorMat = np.resize(colorMat,(colorHeight, colorWidth, 3))
      elif colorFrame.format() == OB_PY_FORMAT_RGB888:
          # Resize the frame data to (height,width,3)
          colorMat = np.resize(colorMat,(colorHeight, colorWidth, 3))
          # Convert frame data RGB to BGR
          colorMat = cv2.cvtColor(colorMat, cv2.COLOR_RGB2BGR)
      elif colorFrame.format() == OB_PY_FORMAT_YUYV:
          colorMat = np.resize(colorMat,(colorHeight, colorWidth, 2))
          # Converting data frame YUYV to RGB data
          colorMat = cv2.cvtColor(colorMat, cv2.COLOR_YUV2BGR_YUYV)
      elif colorFrame.format() == OB_PY_FORMAT_UYVY:
          colorMat = np.resize(colorMat,(colorHeight, colorWidth, 2))
          # Converting data frame YUYV to RGB data
          colorMat = cv2.cvtColor(colorMat, cv2.COLOR_YUV2BGR_UYVY)
      elif colorFrame.format() == OB_PY_FORMAT_I420:
          colorMat = colorMat.reshape((colorHeight * 3 // 2, colorWidth))
          colorMat = cv2.cvtColor(colorMat, cv2.COLOR_YUV2BGR_I420)
          colorMat = cv2.resize(colorMat, (colorWidth, colorHeight))
          
      # print("color width = %d height = %d\n" %(colorFrame.width(),colorFrame.height()))
      # Convert frame data BGR to RGB
      if colorMat is not None:
        colorMat = cv2.cvtColor(colorMat,cv2.COLOR_BGR2RGB)
        # Save color chart raw data as raw format
        colorMat.tofile(colorRawName)
        newData = colorMat
        # Resize the frame data to (height,width,3)
        newData = np.resize(newData,(colorFrame.height(), colorFrame.width(), 3))
        # Convert frame data RGB to BGR
        colorMat = cv2.cvtColor(colorMat,cv2.COLOR_RGB2BGR)
        # Save color image in png format
        cv2.imwrite(colorPngName,colorMat)

    if depthFrame !=  None and depthCount < 5:
      # Save depth images
      depthCount += 1
      depthRawName = "depth_" + str(depthFrame.timeStamp()) + ".raw"
      depthPngName = "depth_" + str(depthFrame.timeStamp()) + ".png"
      data = depthFrame.data()
      # Resize the frame data to (height,width,2)
      data = np.resize(data,(depthFrame.height(), depthFrame.width(), 2))
      # Convert frame data 8bit to 16bit
      newData = data[:,:,0]+data[:,:,1]*256
      # Convert frame data to 1mm unit
      newData = (newData * depthFrame.getValueScale()).astype('uint16')
      
      # Save depth chart raw data as raw format
      newData.tofile(depthRawName)
      # Save depth image in png format
      cv2.imwrite(depthPngName, newData)

  # stop pipeline
  pipe.stop()

except ObException as e:
  print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))