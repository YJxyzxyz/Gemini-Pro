from ObTypes import *
from Property import *
import Pipeline
import StreamProfile
from Error import ObException
import cv2
import numpy as np
import sys

#  Convert frame data from 16bit to 8bit
def mapUint16ToUint8(img, lowerBound = None, upperBound = None):
  if lowerBound == None:
    lowerBound = np.min(img)
  if upperBound == None:
    upperBound = np.max(img)
  lut = np.concatenate([
    np.zeros(lowerBound, dtype=np.uint16),
    np.linspace(0, 255, upperBound - lowerBound).astype(np.uint16),
    np.ones(2**16 - upperBound, dtype = np.uint16) * 255
  ])
  return lut[ img ].astype(np.uint8)

q = 113
ESC = 27

try:
  # Create a Pipeline, which is the entry point for the entire advanced API 
  # and can be easily opened and closed through the Pipeline Multiple types of streams and get a set of frame data
  pipe = Pipeline.Pipeline(None, None)
  # Configure which streams to enable or disable in Pipeline by creating a Config
  config = Pipeline.Config()

  windowsWidth = 0
  windowsHeight = 0
  try:
    # Get all stream configurations for the IR camera, including the stream resolution, frame rate, and frame format
    profiles = pipe.getStreamProfileList(OB_PY_SENSOR_IR)

    videoProfile = None
    try:
      # Select the default resolution to open the stream, you can configure the default resolution through the configuration file.
      videoProfile = profiles.getProfile(0)
    except ObException as e:
      print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
      
    depthProfile = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
    windowsWidth = depthProfile.width()
    windowsHeight = depthProfile.height()
    config.enableStream(depthProfile)
  except ObException as e:
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
    print("Current device is not support IR sensor!")
    sys.exit()

  # Start the stream configured in Config, if no parameters are passed, the default configuration will be used.
  pipe.start(config, None)

  # Get whether the mirror property has writable permissions.
  if pipe.getDevice().isPropertySupported(OB_PY_PROP_IR_MIRROR_BOOL, OB_PY_PERMISSION_WRITE):
    # Set Mirror.
    pipe.getDevice().setBoolProperty(OB_PY_PROP_IR_MIRROR_BOOL, True)

  while True:
    # wait in a blocking manner for a frame that is a composite frame containing frame data
    # for all the streams enabled in the configuration and set the frame wait timeout to 100ms
    frameSet = pipe.waitForFrames(100)
    if frameSet == None:
      continue
    else:
      # Render a set of frame data in the window, here only the IR frames are rendered
      irFrame = frameSet.irFrame()
      if irFrame != None:
        size = irFrame.dataSize()
        data = irFrame.data()
        ir_format = irFrame.format()
        if size != 0:
          # Resize the frame data to (height,width,2)
          if ir_format == int(OB_PY_FORMAT_Y16):
            data = np.resize(data,(windowsHeight, windowsWidth, 2))
            # Convert frame data from 8bit to 16bit
            newData = data[:,:,0]+data[:,:,1]*256
          elif ir_format == int(OB_PY_FORMAT_Y8):
            data = np.resize(data,(windowsHeight, windowsWidth, 1))
            # Convert frame data from 8bit to 16bit
            newData = data[:, :, 0]
          # Convert frame data from 16bit to 8bit for rendering
          newData = mapUint16ToUint8(newData)
          # Convert frame data GRAY to RGB
          newData = cv2.cvtColor(newData, cv2.COLOR_GRAY2RGB) 

          cv2.namedWindow("InfraredViewer", cv2.WINDOW_NORMAL)

          cv2.imshow("InfraredViewer", newData)

          key = cv2.waitKey(1)
          # Press ESC or 'q' to close the window
          if key == ESC or key == q:
            cv2.destroyAllWindows()
            break

  # Stopping Pipeline will no longer generate frame data
  pipe.stop()

except ObException as e:
  print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))