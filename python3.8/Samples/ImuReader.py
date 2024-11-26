from ObTypes import *
from Property import *
import Context
import Pipeline
import StreamProfile
import Version
from Error import ObException
import cv2
import sys
import platform

plat = platform.system().lower()
if plat == 'windows':
  import msvcrt
elif plat == 'linux':
  import termios, tty, select
  oldSettings = termios.tcgetattr(sys.stdin)
  tty.setcbreak(sys.stdin.fileno())

ESC = 27

def gyroCallback(frame):
  timeStamp = frame.timeStamp()
  gyroFrame = frame.toConcreteFrame(OB_PY_FRAME_GYRO)
  if gyroFrame != None and (timeStamp % 500) < 2:  #( timeStamp % 500 ) < 2: The purpose is to reduce the frequency of printing
    value = gyroFrame.value()
    print("Gyro Frame: tsp = %d, temperature = %f, gyro.x = %f dps, gyro.y = %f dps, gyro.z = %f dps\n" %(timeStamp, gyroFrame.temperature(), value.get("x"), value.get("y"), value.get("z")))

def accelCallback(frame):
  timeStamp = frame.timeStamp()
  accelFrame = frame.toConcreteFrame(OB_PY_FRAME_ACCEL)
  if accelFrame != None and (timeStamp % 500) < 2:  # ( timeStamp % 500 ) < 2: The purpose is to reduce the frequency of printing
    value = accelFrame.value()
    print("Accel Frame: tsp = %d, temperature = %f, accel.x = %f g, accel.y = %f g, accel.z = %f g\n" %(timeStamp, accelFrame.temperature(), value.get("x"), value.get("y"), value.get("z")))

try:
  # Print the SDK version number, which is divided into the main version number, the secondary version number and the revision number
  version = Version.Version()
  print("SDK version:%d.%d.%d" %(version.getMajor(), version.getMinor(), version.getPatch()))

  # Create a Context, unlike Pipeline, Context is the entrance to the underlying API, which is slightly more complicated
  # to use for common operations such as switching streams, but the underlying API can provide more flexible operations, 
  # such as getting multiple devices, reading and writing device and camera properties, etc.
  ctx = Context.Context(None)

  ctx.setLoggerSeverity(OB_PY_LOG_SEVERITY_NONE)

  # Query the list of already connected devices  
  devList = ctx.queryDeviceList()

  # Get the number of access devices
  if devList.deviceCount() == 0:
    print("Device not found!")
    sys.exit()

  # Create device, 0 means the index of the first device
  dev = devList.getDevice(0)
  gyroSensor = None
  accelSensor = None
  try:
  # Get Gyroscope Sensor
    gyroSensor = dev.getSensorList().getSensorByType(OB_PY_SENSOR_GYRO)
    if gyroSensor != None:
      # Get the configuration list
      profiles = gyroSensor.getStreamProfileList()
      # Select the first configuration to open the stream
      profile = profiles.getProfile(0)
      gyroSensor.start(profile, gyroCallback)
    else:
      print("get Gyro Sensor failed !")

    # Get Accelerometer Sensor
    accelSensor = dev.getSensorList().getSensorByType(OB_PY_SENSOR_ACCEL)
    if accelSensor != None:
      # Get the configuration list
      profiles = accelSensor.getStreamProfileList()
      # Select the first configuration to open the stream
      profile = profiles.getProfile(0)
      accelSensor.start(profile, accelCallback)
    else:
      print("get Accel Sensor failed !")

  except ObException as e:
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
    print("current device is not support imu!")

  print("Press ESC to exit!")
  while True:
    # Press ESC key to exit
    if plat == 'windows':
      if msvcrt.kbhit():
        key = ord(msvcrt.getch())
        if key == ESC:
          print("stop")
          break	
    elif plat == 'linux':
      if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        key = sys.stdin.read(1)
        if key == '\x1b':
          print("stop")
          break
        sys.stdout.write(key)
        sys.stdout.flush()

  # Close stream
  if gyroSensor != None:
    gyroSensor.stop()

  if accelSensor != None:
    accelSensor.stop()

  if plat == 'linux':
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, oldSettings)

except ObException as e:
  print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))