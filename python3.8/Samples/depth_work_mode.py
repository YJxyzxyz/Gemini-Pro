from ObTypes import *
from Property import *
import Pipeline
from Property import *
import StreamProfile
from Error import ObException
import cv2
import numpy as np
import sys

q = 113
ESC = 27


def toHexText(data, dataLen):
    alpha = "0123456789abcdef"
    hexStr = bytearray(dataLen * 2 + 1)
    for i in range(dataLen):
        val = data[i]
        hexStr[i] = alpha[(val >> 4) & 0xf]
        hexStr[i + 1] = alpha[val & 0xf]
    return hexStr.decode()


def main():
    # Create a Pipeline, which is the entry point for the entire Advanced API and can be easily opened and closed through the Pipeline
    # Multiple types of streams and acquire a set of frame data
    pipeline = Pipeline.Pipeline(None, None)
    device = pipeline.getDevice()
    if not device.isPropertySupported(OB_PY_STRUCT_CURRENT_DEPTH_ALG_MODE, OB_PY_PERMISSION_READ_WRITE):
        print("depth work mode is unsupported")
        return
    cur_work_mode = device.getCurrentDepthWorkMode()
    print("current work mode is %s" % cur_work_mode.get("name"))
    mode_list = device.getDepthWorkModeList()
    mode_count = mode_list.count()
    print("Support depth work mode list count: %d\n" % mode_count)
    for i in range(mode_count):
        mode = mode_list.getDepthWorkMode(i)
        print("mode %d: %s\n" % (i, mode.get("name")))
        if cur_work_mode.get("name") == mode.get("name"):
            print("(current work mode)")
        print("\n")
    mode_index = int(input("Please input the index of work mode: "))
    if mode_index < 0 or mode_index >= mode_count:
        print("Invalid index\n")
        return
    mode = mode_list.getDepthWorkMode(mode_index)
    print("Set work mode to %s" % mode.get("name"))
    device.switchDepthWorkMode(mode.get("name"))
    print("Set work mode to %s success" % mode.get("name"))
    cur_work_mode = device.getCurrentDepthWorkMode()
    print("current work mode is %s" % cur_work_mode.get("name"))
    print("Press ESC to exit! \n")
    while True:
        key = cv2.waitKey(1)
        if key == ESC or key == q:
            break
        else:
            continue

    pipeline.stop()

if __name__ == "__main__":
    main()
