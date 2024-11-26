from ObTypes import *
from Property import *
import Pipeline
import RecordPlayback
import StreamProfile
from Error import ObException
import cv2
import numpy as np
import sys

lowApi = True

q = 113
ESC = 27

try:
  # Create a Pipeline, which is the entry point for the entire advanced API.
  # Through the Pipeline it is easy to open and close multiple types of streams and get a set of frame data
  pipe = Pipeline.Pipeline(None, None)

  # Configure which streams to enable or disable in Pipeline by creating a Config
  config = Pipeline.Config()

  windowsWidth = 0
  windowsHeight = 0
  depthProfile = None
  
  try:
    # Get all stream configurations for the depth camera, including the stream resolution, frame rate, and frame format
    profiles = pipe.getStreamProfileList(OB_PY_SENSOR_DEPTH)

    videoProfile = None
    try:
      # Find the corresponding profile according to the specified format, preferring Y16 format
      videoProfile = profiles.getVideoStreamProfile(640,0,OB_PY_FORMAT_Y16,30)
    except ObException as e:
      print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
      # After not finding Y16 format does not match the format to find the corresponding Profile to open the stream
      videoProfile = profiles.getVideoStreamProfile(640,0,OB_PY_FORMAT_UNKNOWN,30)

    depthProfile  = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
    windowsWidth  = depthProfile.width()
    windowsHeight = depthProfile.height()
  except ObException as e:
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
    print("Current device is not support depth sensor!")
    sys.exit()

  config.enableStream(depthProfile)

  # Start the stream configured in Config, if no parameters are passed, it will start the default configuration start stream
  pipe.start(config, None)

  pipe.startRecord(b'./OrbbecPipeline.bag')

  while True:
    # Wait for a frame in a blocking manner, which is a composite frame containing the 
    # frame data of all streams enabled in the configuration, and set the frame wait timeout to 100ms
    frameSet = pipe.waitForFrames(100)
    if frameSet == None:
      continue
    else:
      # Renders a set of frames in the window, here only the depth frames are rendered
      depthFrame = frameSet.depthFrame()
      if depthFrame != None:
        size = depthFrame.dataSize()
        data = depthFrame.data()
        valueScale = depthFrame.getValueScale()
        if size != 0:
          # Resize the frame data to (height,width,2)
          data = np.resize(data,(depthFrame.height(), depthFrame.width(), 2))
          # Convert frame data from 8bit to 16bit
          newData = data[:,:,0]+data[:,:,1]*256          
          # Conversion of frame data to 1mm units
          newData = (newData * valueScale).astype('uint16')
          # Rendering display
          newData = (newData / 32).astype('uint8')
          # Convert frame data GRAY to RGB
          newData = cv2.cvtColor(newData, cv2.COLOR_GRAY2RGB) 

          # Create Window
          cv2.namedWindow("Recorder", cv2.WINDOW_NORMAL)

          # Display image
          cv2.imshow("Recorder", newData)

          key = cv2.waitKey(1)
          # Press ESC or 'q' to close the window
          if key == ESC or key == q:
            cv2.destroyAllWindows()
            break
  pipe.stopRecord()
  # Stopping Pipeline will no longer generate frame data
  pipe.stop()

except ObException as e:
  print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))