from ObTypes import *
import Context
import Pipeline
import StreamProfile
from Error import ObException
import sys
import platform

plat = platform.system().lower()
if plat == 'windows':
  import msvcrt
elif plat == 'linux':
  import termios, tty, select
  oldSettings = termios.tcgetattr(sys.stdin)
  tty.setcbreak(sys.stdin.fileno())

pipeline = None

ESC = 27
r = 114
R = 82

def CreateAndStartWithConfig():
  global pipeline
  if pipeline != None:
    try:
      # Configure the startup stream by stream of the configuration file,
      # if there is no configuration file, the 0th stream will be used to configure the startup stream
      pipeline.start(None, None)
    except ObException as e:
      print("Pipeline start failed!")
      print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))	

#  Device connection callback
def DeviceConnectCallback(connectList):
  global pipeline
  print("Device connect: %d",connectList.deviceCount())
  if connectList.deviceCount() > 0:
    if pipeline == None:
      pipeline = Pipeline.Pipeline(None, None)
      CreateAndStartWithConfig()

#  Device disconnect callback
def DeviceDisconnectCallback(disconnectList):
  global pipeline
  print("Device disconnect: %d",disconnectList.deviceCount())
  if disconnectList.deviceCount() > 0 :		
    if pipeline != None:
      try :
        pipeline.stop()
      except ObException as e:
        print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))	
      pipeline = None

def callback(removedList, addedList):
  if removedList != None:
    DeviceDisconnectCallback(removedList)
  if addedList != None:
    DeviceConnectCallback(addedList)
  print("hotplug event")

def getFrame(pipeline):
  if pipeline != None:
    # Wait for a frame of data with a timeout of 100ms
    frameSet = pipeline.waitForFrames(100)
    if frameSet != None:
      depthFrame = frameSet.depthFrame()
      if depthFrame != None:
        pixelValue = 0
        size = depthFrame.dataSize()
        data = depthFrame.data()
        if size != 0:
          width      = depthFrame.width()
          height     = depthFrame.height()
          data.resize((height, width, 2))
          newData = data[:,:,0]+data[:,:,1]*256
          pixelValue = newData[int(height/2),int(width/2)]

        print("=====Depth Frame Info======\n FrameType: %d ,index: %d ,width: %d ,height: %d ,format: %d ,timeStampUs: %d ,middlePixelValue: %d"\
          %(depthFrame.type(),depthFrame.index(),depthFrame.width(),depthFrame.height(),\
          depthFrame.format(),depthFrame.timeStampUs(),pixelValue))

      colorFrame = frameSet.colorFrame()
      if colorFrame != None:
        print("=====Color Frame Info======\n FrameType: %d ,index: %d ,width: %d ,height: %d ,format: %d ,timeStampUs: %d"\
          %(colorFrame.type(),colorFrame.index(),colorFrame.width(),colorFrame.height(),\
          colorFrame.format(),colorFrame.timeStampUs()))

try:
  ctx = Context.Context(None)
  ctx.setDeviceChangedCallback(callback)
  devList = ctx.queryDeviceList()
  if devList.deviceCount() > 0:
    if pipeline == None:
      pipeline = Pipeline.Pipeline(None, None)
      CreateAndStartWithConfig()

  while True:
    if plat == 'windows':
      if msvcrt.kbhit():
        key = ord(msvcrt.getch())
        # Press ESC key to exit
        if key == ESC:
          print("stop")
          break	
        # Press r to reboot the device to trigger the device offline/online,
        # you can also manually unplug the device to trigger	
        elif key == r or key == R:
          if pipeline != None:
            try:
              pipeline.getDevice().reboot()
            except ObException as e:
              print("device reboot failed!")
              print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))	

    elif plat == 'linux':
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
          key = sys.stdin.read(1)
          print("key=%s" %key)
          if key == '\x1b':
            print("stop")
            break
          # Press r to reboot the device to trigger the device offline/online,
          # you can also manually unplug the device to trigger	
          elif key == 'r' or key == 'R':
            if pipeline != None:
              try:
                pipeline.getDevice().reboot()
              except ObException as e:
                print("device reboot failed!")
                print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))	

          sys.stdout.write(key)
          sys.stdout.flush()

    getFrame(pipeline)

  if plat == 'linux':
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, oldSettings)

except ObException as e:
  print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))