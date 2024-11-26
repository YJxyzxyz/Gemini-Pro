from ObTypes import *
import Version
import Context
import Device
import Sensor
from Error import ObException
import sys
import platform

plat = platform.system().lower()
if plat == 'windows':
  import msvcrt
elif plat == 'linux':
  import termios, tty

ESC = 27

try:
  # Print the SDK version number, which is divided into the main version number, the secondary version number and the revision number
  version= Version.Version()
  print("SDK version: %d.%d.%d" \
    %(version.getMajor(), version.getMinor(), version.getPatch()))

  # Create a Context, unlike Pipeline, the Context is the entry point for the underlying API, 
  # in terms of common operations such as switching streams Using low-level will be slightly 
  # more complicated, but the underlying API can provide more flexible operations, such as getting
  # multiple devices, reading and writing properties of devices and cameras, etc.
  ctx = Context.Context(None)

  # Query the list of already connected devices
  devList = ctx.queryDeviceList()

  # Get the number of access devices
  devCount = devList.deviceCount()
  if devCount == 0:
    print("Device not found!")
    sys.exit()

  # Create device, 0 means the index of the first device
  dev = devList.getDevice(0)

  # Obtain device information
  devInfo = dev.getDeviceInfo()

  # Get the name of the device
  print("Device name:: %s" %(devInfo.name()))

  # Get the PID, VID, and UID of the device
  print("Device pid: %#x vid: %#x uid: 0x%s" \
    %(devInfo.pid(), devInfo.vid(), devInfo.uid()))

  # Get the firmware version number of the device
  fwVer = devInfo.firmwareVersion()
  print("Firmware version: %s" %(fwVer))

  # Get the serial number of the device
  sn = devInfo.serialNumber()
  print("Serial number: %s" %(sn))

  # Get a list of supported sensors
  print("Sensor types: ")

  sensors = dev.getSensorList()
  for i in range(sensors.count()):
    sensor = sensors.getSensorByIndex(i)
    sensorType = sensor.type()
    if sensorType == OB_PY_SENSOR_COLOR :
      print("\tColor sensor")
    elif sensorType == OB_PY_SENSOR_DEPTH :
      print("\tDepth sensor")
    elif sensorType == OB_PY_SENSOR_IR :
      print("\tIR sensor")
    elif sensorType == OB_PY_SENSOR_GYRO :
      print("\tGyro sensor")
    elif sensorType == OB_PY_SENSOR_ACCEL :
      print("\tAccel sensor")
    else:
      print("\tUNKNOWN")

  print("Press ESC to exit! ")
  while True:
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

except ObException as e:
  print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))