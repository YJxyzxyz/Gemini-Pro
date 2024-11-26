from ObTypes import *
from Property import *
import Pipeline
import StreamProfile
from Error import ObException
import cv2
import numpy as np
import sys
import math

q = 113
ESC = 27

try:
  # Create a Pipeline, which is the entry point for the entire Advanced API and can be easily opened and closed through the Pipeline
  # Multiple types of streams and acquire a set of frame data
  pipe = Pipeline.Pipeline(None, None)
  # Configure which streams to enable or disable in Pipeline by creating a Config
  config = Pipeline.Config()

  windowsWidth = 0
  windowsHeight = 0
  try:
    # Get all stream configurations for the depth camera, including the stream resolution, frame rate, and frame format
    profiles = pipe.getStreamProfileList(OB_PY_SENSOR_DEPTH)

    videoProfile = None
    try:
      # Select the default resolution to open the stream, you can configure the default resolution through the configuration file
      videoProfile = profiles.getProfile(0)
    except ObException as e:
      print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
      
    depthProfile = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
    windowsWidth = depthProfile.width()
    windowsHeight = depthProfile.height()
    config.enableStream(depthProfile)
  except ObException as e:
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
    print("Current device is not support depth sensor!")
    sys.exit()

  # Start the stream configured in Config, if no parameters are passed, it will start the default configuration start stream.
  pipe.start(config, None)

  # Get whether the mirror property has writable permissions
  if pipe.getDevice().isPropertySupported(OB_PY_PROP_DEPTH_MIRROR_BOOL, OB_PY_PERMISSION_WRITE):
    # Set Mirror
    pipe.getDevice().setBoolProperty(OB_PY_PROP_DEPTH_MIRROR_BOOL, True)

  while True:
    # Waiting in a blocking manner for a frame of data, which is a composite frame containing frame data for all streams enabled in the configuration.
    # and set the frame wait timeout to 100ms
    frameSet = pipe.waitForFrames(100)
    if frameSet == None:
      continue
    else:
      # Renders a set of frames in the window, here only the depth frames are rendered
      depthFrame = frameSet.depthFrame()
      if depthFrame != None:
        size = depthFrame.dataSize()
        data = depthFrame.data()
        if size != 0:
          # Resize the frame data to (height,width,2)
          data = np.resize(data,(windowsHeight, windowsWidth, 2))
          
          # Convert frame data from 8bit to 16bit
          newData = data[:,:,0]+data[:,:,1]*256          
          # Conversion of frame data to 1mm units
          newData = (newData * depthFrame.getValueScale()).astype('uint16')
          # Rendering display
          newData = (newData / 32).astype('uint8')
          # Convert frame data GRAY to RGB
          newData = cv2.cvtColor(newData, cv2.COLOR_GRAY2RGB) 

          cv2.namedWindow("DepthViewer", cv2.WINDOW_NORMAL)

          cv2.imshow("DepthViewer", newData)

          key = cv2.waitKey(1)
          if key == ESC or key == q:
            cv2.destroyAllWindows()
            break
  pipe.stop()

except ObException as e:
  print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))