from ObTypes import *
from Property import *
import Context
import Device
from Error import ObException
import re

# Determines if the input string is a reasonable range of integers or bool [min,max)
def isCorrectInt(num, min, max):
  value = re.compile(r'^[-+]?[0-9]+$')
  result = value.match(num)
  if result:
    if int(num) < min or int(num) >= max:
      return False
    else:
      return True
  else:
    return False

# Determine if the input string is a reasonable range of Float  [min,max]
def isCorrectFloat(num, min, max):
  value = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
  result = value.match(num)
  if result:
    if float(num) < min or float(num) > max:
      return False
    else:
      return True
  else:
    return False

def permissionTypeToString(permission):
  if permission == OB_PY_PERMISSION_READ:
    return "R/_"
  elif permission == OB_PY_PERMISSION_WRITE:
    return "_/W"
  elif permission == OB_PY_PERMISSION_READ_WRITE:
    return "R/W"
  else:
    return "_/_"

# Select a device, where the device name, pid, vid, uid will be printed,
# and the corresponding device object will be created after selection
def selectDevice(deviceList) :
  devCount = deviceList.deviceCount()
  print("Device list: ")
  for i in range(devCount):
    print("%d. name:%s, vid: %#x, pid: %#x, uid: 0x%s , sn: %s" \
      %(i, deviceList.name(i), deviceList.vid(i), deviceList.pid(i), deviceList.uid(i), deviceList.serialNumber(i)))

  inputInfo = input("Select a device: ")
  while isCorrectInt(inputInfo, 0, devCount) == False:
    inputInfo = input("Your select is out of range, please reselect: ")
  devIndex = int(inputInfo)
  print(devIndex)
  return deviceList.getDevice(devIndex)

# Print the list of supported properties
def printfPropertyList(device):
  size = device.getSupportedPropertyCount()
  if size == 0:
    print("No supported property!")
  print("------------------------------------------------------------------------")
  index = 0
  for i in range(size):
    propertyItem = device.getSupportedProperty(i)
    if propertyItem.get("type") != OB_PY_STRUCT_PROPERTY and propertyItem.get("permission") != OB_PY_PERMISSION_DENY:
      intRange = None
      floatRange = None
      boolRange = None
      if propertyItem.get("type") == OB_PY_BOOL_PROPERTY:
        strRange = "Bool value(min:0, max:1, step:1)"
      elif propertyItem.get("type") == OB_PY_INT_PROPERTY:
        try :
          intRange = device.getIntPropertyRange(propertyItem.get("id"))
          strRange  = "Int value(min:" + str(intRange.get("min")) + ", max:" + str(intRange.get("max")) + ", step:" + str(intRange.get("step")) + ")"
        except ObException as e:
          print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
          print("get int property range failed.")
      elif propertyItem.get("type") == OB_PY_FLOAT_PROPERTY:
        try :
          floatRange = device.getFloatPropertyRange(propertyItem.get("id"))
          strRange  = "Float value(min:" + str(floatRange.get("min")) + ", max:" + str(floatRange.get("max")) + ", step:" + str(floatRange.get("step")) + ")"
        except ObException as e:
          print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
          print("get int property range failed.")
      else:
        print("get property type unknow.")

      print("%d : %s , permission= %s , range= %s" %(index, propertyItem.get("name"), permissionTypeToString(propertyItem.get("permission")), strRange))
      index += 1
  print("------------------------------------------------------------------------")

# Get a list of properties
def getPropertyList(device):
  propertyVec = []
  size = device.getSupportedPropertyCount()
  for i in range(size):
    propertyItem = device.getSupportedProperty(i)
    if propertyItem.get("type") != OB_PY_STRUCT_PROPERTY and propertyItem.get("permission") != OB_PY_PERMISSION_DENY:
      propertyVec.append(propertyItem)
  return propertyVec

# Set Properties
def setPropertyValue(device, propertyItem,strValue):
  try :
    intValue   = 0
    floatValue = 0
    boolValue  = 0
    if propertyItem.get("type") == OB_PY_BOOL_PROPERTY:
      boolValue = int(strValue)
      try :
        device.setBoolProperty(propertyItem.get("id"), boolValue)
      except ObException as e:
        print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
        print("set bool property fail.")
      print("property name: %s ,set bool value: %d" %(propertyItem.get("name"), boolValue))
    elif propertyItem.get("type") == OB_PY_INT_PROPERTY:
      intValue = int(strValue)
      try :
        device.setIntProperty(propertyItem.get("id"), intValue)
      except ObException as e:
        print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
        print("set int property fail.")
      print("property name: %s ,set int value: %d" %(propertyItem.get("name"), intValue))
    elif propertyItem.get("type") == OB_PY_FLOAT_PROPERTY:
      floatValue = float(strValue)
      try :
        device.setFloatProperty(propertyItem.get("id"), floatValue)
      except ObException as e:
        print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
        print("set int property fail.")
      print("property name: %s ,set float value: %f" %(propertyItem.get("name"), floatValue))
    else:
      print("set property type unknow.")

  except ObException as e:
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
    print("set property failed: %s" %(propertyItem.get("name")))

# Get Properties
def getPropertyValue(device, propertyItem):
  try :
    boolRet   = 0
    intRet    = 0
    floatRet  = 0
    if propertyItem.get("type") == OB_PY_BOOL_PROPERTY:
      try :
        boolRet = device.getBoolProperty(propertyItem.get("id"))
      except ObException as e:
        print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
        print("get bool property failed.")
      print("property name: %s ,get bool value: %d" %(propertyItem.get("name"), boolRet))
    elif propertyItem.get("type") == OB_PY_INT_PROPERTY:
      try :
        intRet = device.getIntProperty(propertyItem.get("id"))
      except ObException as e:
        print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
        print("get bool property failed.")
      print("property name: %s ,get int value: %d" %(propertyItem.get("name"), intRet))
    elif propertyItem.get("type") == OB_PY_FLOAT_PROPERTY:
      try :
        floatRet = device.getFloatProperty(propertyItem.get("id"))
      except ObException as e:
        print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
        print("get bool property failed.")
      print("property name: %s ,get float value: %f" %(propertyItem.get("name"), floatRet))

    else:
      print("get property type unknow.")

  except ObException as e:
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
    print("get property failed: %s" %(propertyItem.get("name")))

try:
  # main
  # Create a Context, unlike Pipeline, the Context is the entry point for the underlying API, 
  # in terms of common operations such as switching streams It is slightly more complicated 
  # to use the lower level, but the underlying API can provide more flexible operations, such
  # as getting multiple devices, reading and writing properties of devices and cameras, etc.
  context = Context.Context(None)

  # Query the list of already connected devices
  deviceList = context.queryDeviceList()

  isSelectDevice = True
  while isSelectDevice == True:
    # Select a device for operation
    device = None
    if deviceList.deviceCount() > 0:
      if deviceList.deviceCount() <= 1:
        device = deviceList.getDevice(0)
      else:
        device = selectDevice(deviceList)
      deviceInfo = device.getDeviceInfo()
      print("------------------------------------------------------------------------")
      print("Current Device:\n name: %s , vid: %#x , pid: %#x, uid: 0x%s" \
        %(deviceInfo.name(), deviceInfo.vid(), deviceInfo.pid(),deviceInfo.uid()))
    else:
      print("Device Not Found")
      isSelectDevice = False
      break

    print("Input \"?\" to get all properties.")
    isSelectProperty = True
    while isSelectProperty == True:
      choice = input("")
      if choice != "?":
        controlVec = choice.split()
        if len(controlVec) <= 0:
          continue
        if controlVec[0] == "exit":
          isSelectProperty = False
          isSelectDevice   = False
          break
        # Determine if the input format matches
        if len(controlVec)<= 1 or (controlVec[1] != "get" and controlVec[1] != "set") or len(controlVec) > 3\
          or (controlVec[1] == "set" and len(controlVec) < 3):
          print("Property control usage: [property index] [set] [property value] or [property index] [get]")
          continue

        propertyList = getPropertyList(device)
        size		 = len(propertyList)
        selectId	 = int(controlVec[0])
        if selectId >= size:
          print("Your selection is out of range, please reselect: ")
          continue
        isGetValue = False
        if controlVec[1] == "get":
          isGetValue = True
        else:
          isGetValue = False

        propertyItem = propertyList[selectId]

        if isGetValue == True:
          # Get property values
          getPropertyValue(device, propertyItem)
        else:
          # Set property value
          setPropertyValue(device, propertyItem, controlVec[2])
      else:
        printfPropertyList(device)
        print("Please select property.(Property control usage: [property number] [set/get] [property value])")

except ObException as e:
  print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" %(e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))