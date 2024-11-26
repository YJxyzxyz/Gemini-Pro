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

mediaState = None
frameSet = None

def mediaStateCallback(state):
  global mediaState
  if state == OB_PY_MEDIA_BEGIN:
    print("Playback file begin.")
  elif state == OB_PY_MEDIA_END:
    print("Playback file end.")
    mediaState = OB_PY_MEDIA_END

def playbackStartCallback(frame):
  global frameSet
  frameSet = frame
  print("playbackStartCallback")

try:
  # Create a pipeline object for playback
  pipe = Pipeline.Pipeline(None, b'./OrbbecPipeline.bag')

  # Get playback object
  playback = pipe.getPlayback()
  # Set playback status callback
  playback.setPlaybackStateCallback(mediaStateCallback)

  # Read device information from playback files
  deviceInfo = playback.getDeviceInfo()
  print("======================DeviceInfo: name : %s sn: %s firmware: %s vid: %#x pid: %#x" 
    %(deviceInfo.name(), deviceInfo.serialNumber(), deviceInfo.firmwareVersion(), deviceInfo.vid(), deviceInfo.pid()))

  # Read internal reference information from playback files
  cameraParam = pipe.getCameraParam()
  print("======================Camera params : rgb width: %d rgb height: %d depth width: %d depth height: %d"\
    %(cameraParam.get("rgbIntrinsic").get("width"),cameraParam.get("rgbIntrinsic").get("height"),\
    cameraParam.get("depthIntrinsic").get("width"), cameraParam.get("depthIntrinsic").get("height")))

  # start Playback
  pipe.start(None, None)

  while True:
    if mediaState == OB_PY_MEDIA_END:
      break
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
        width = depthFrame.width()
        height = depthFrame.height()
        valueScale = depthFrame.getValueScale()
        if size != 0:
          # Resize the frame data to (height,width,2)
          data = np.resize(data,(height, width, 2))
          # Convert frame data from 8bit to 16bit
          newData = data[:,:,0]+data[:,:,1]*256          
          # Conversion of frame data to 1mm units
          newData = (newData * valueScale).astype('uint16')
          # Rendering display
          newData = (newData / 32).astype('uint8')
          # Convert frame data GRAY to RGB
          newData = cv2.cvtColor(newData, cv2.COLOR_GRAY2RGB) 

          # Create Window
          cv2.namedWindow("Playback", cv2.WINDOW_NORMAL)

          # Display image
          cv2.imshow("Playback", newData)

          key = cv2.waitKey(1)
          # Press ESC or 'q' to close the window
          if key == ESC or key == q:
            cv2.destroyAllWindows()
            break
  # stop pipeline
  pipe.stop()

except ObException as e:
  print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))