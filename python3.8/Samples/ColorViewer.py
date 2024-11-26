from ObTypes import *
from Property import *
import Pipeline
import StreamProfile
from Error import ObException
import cv2
import numpy as np
import sys

q = 113
ESC = 27

try:
  # Create a Pipeline, which is the entry point for the entire Advanced API.
  pipe = Pipeline.Pipeline(None, None)
  # Configure which streams to enable or disable in Pipeline.
  config = Pipeline.Config()

  windowsWidth = 0
  windowsHeight = 0
  try:
    # Get all stream configurations for the color camera, including the stream resolution, frame rate, and frame format
    profiles = pipe.getStreamProfileList(OB_PY_SENSOR_COLOR)

    videoProfile = None
    try:
      # Select the first to open the stream.
      videoProfile = profiles.getProfile(0)
    except ObException as e:
      print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))

    colorProfile = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
    windowsWidth = colorProfile.width()
    windowsHeight = colorProfile.height()
    config.enableStream(colorProfile)
  except ObException as e:
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
    print("Current device is not support color sensor!")
    sys.exit()

  # Start the stream configured in Config, if no parameters are passed, it will start the default configuration start stream
  pipe.start(config, None)

  # Get whether the mirror property has writable permissions
  if pipe.getDevice().isPropertySupported(OB_PY_PROP_COLOR_MIRROR_BOOL, OB_PY_PERMISSION_WRITE):
    # Set Mirror
    pipe.getDevice().setBoolProperty(OB_PY_PROP_COLOR_MIRROR_BOOL, True)

  while True:
    # Waiting for a frame in a blocking manner, which is a composite frame containing frame data for all enabled streams in the configuration.
    # And setting the frame's wait timeout to 100ms.
    frameSet = pipe.waitForFrames(100)
    if frameSet == None:
      continue
    else:
      # Render a set of frame data in a window, rendering only color frames here.
      colorFrame = frameSet.colorFrame()
      if colorFrame != None:
        # To get the size and data of a frame:
        size = colorFrame.dataSize()
        data = colorFrame.data()

        if size != 0:
          newData = data
          if colorFrame.format() == OB_PY_FORMAT_MJPG:
              newData = cv2.imdecode(newData,1)
              if newData is not None:
                newData = np.resize(newData,(windowsHeight, windowsWidth, 3))
          elif colorFrame.format() == OB_PY_FORMAT_RGB888:
              newData = np.resize(newData,(windowsHeight, windowsWidth, 3))
              newData = cv2.cvtColor(newData, cv2.COLOR_RGB2BGR)
          elif colorFrame.format() == OB_PY_FORMAT_YUYV:
              newData = np.resize(newData,(windowsHeight, windowsWidth, 2))
              newData = cv2.cvtColor(newData, cv2.COLOR_YUV2BGR_YUYV)
          elif colorFrame.format() == OB_PY_FORMAT_UYVY:
              newData = np.resize(newData,(windowsHeight, windowsWidth, 2))
              newData = cv2.cvtColor(newData, cv2.COLOR_YUV2BGR_UYVY)
          elif colorFrame.format() == OB_PY_FORMAT_I420:
              newData = newData.reshape((windowsHeight * 3 // 2, windowsWidth))
              newData = cv2.cvtColor(newData, cv2.COLOR_YUV2BGR_I420)
              newData = cv2.resize(newData, (windowsWidth, windowsHeight))
              
           
          cv2.namedWindow("ColorViewer", cv2.WINDOW_NORMAL)

          if newData is not None:
            cv2.imshow("ColorViewer", newData)

          key = cv2.waitKey(1)
          # Press ESC or 'q' to close the window
          if key == ESC or key == q:
            cv2.destroyAllWindows()
            break
          
  pipe.stop()

except ObException as e:
  print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))