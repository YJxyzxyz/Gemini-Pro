from ObTypes import *
import Context
import Pipeline
import StreamProfile
import Device
from Error import ObException
import cv2
import numpy as np
import sys

q = 113
ESC = 27

# This program demonstrates up to 3 devices and can be adjusted by the user as needed
maxDevNum = 3
colorFrames = [None] * maxDevNum
depthFrames = [None] * maxDevNum


def dev0FramesetCallback(frameSet):
    global colorFrames
    global depthFrames
    if frameSet.colorFrame() != None and frameSet.depthFrame() != None:
        colorFrames[0] = frameSet.colorFrame()
        depthFrames[0] = frameSet.depthFrame()


def dev1FramesetCallback(frameSet):
    global colorFrames
    global depthFrames
    if frameSet.colorFrame() != None and frameSet.depthFrame() != None:
        colorFrames[1] = frameSet.colorFrame()
        depthFrames[1] = frameSet.depthFrame()


def dev2FramesetCallback(frameSet):
    global colorFrames
    global depthFrames
    if frameSet.colorFrame() != None and frameSet.depthFrame() != None:
        colorFrames[2] = frameSet.colorFrame()
        depthFrames[2] = frameSet.depthFrame()


# Open depth and color streams for all devices
def StartStream(pipes):
    devIndex = 0
    for pipe in pipes:
        # Create Config
        config = Pipeline.Config()
        try:
            # Get all stream configurations for the color camera, including the stream resolution,
            # frame rate, and frame format
            profiles = pipe.getStreamProfileList(OB_PY_SENSOR_COLOR)

            videoProfile = None
            try:
                # Select the default resolution to open the stream, you can 
                # configure the default resolution through the configuration file
                videoProfile = profiles.getProfile(0)
            except ObException as e:
                print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
                e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
                
            colorProfile = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
            config.enableStream(colorProfile)
        except ObException as e:
            print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
            e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
            print("Current device is not support color sensor!")
            sys.exit()

        try:
            # Get all stream configurations for the depth camera, 
            # including the stream resolution, frame rate, and frame format
            profiles = pipe.getStreamProfileList(OB_PY_SENSOR_DEPTH)

            videoProfile = None
            try:
                # Select the default resolution to open the stream, you can configure the default resolution through the configuration file
                videoProfile = profiles.getProfile(0)
            except ObException as e:
                print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
                e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
                
            depthProfile = videoProfile.toConcreteStreamProfile(OB_PY_STREAM_VIDEO)
            config.enableStream(depthProfile)
        except ObException as e:
            print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
            e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
            print("Current device is not support depth sensor!")
            sys.exit()

        if devIndex == 0:
            pipe.start(config, dev0FramesetCallback)
        if devIndex == 1:
            pipe.start(config, dev1FramesetCallback)
        if devIndex == 2:
            pipe.start(config, dev2FramesetCallback)
        devIndex += 1


try:
    # Main function entry
    pipes = []
    key = -1

    # Create a Context to get a list of devices
    ctx = Context.Context(None)

    # Query the list of already connected devices
    devList = ctx.queryDeviceList()

    # Get the number of access devices
    devCount = devList.deviceCount()

    # Iterate through the list of devices and create a pipeline
    for i in range(devCount):
        # Get the device and create a Pipeline
        dev = devList.getDevice(i)
        pipe = Pipeline.Pipeline(dev, None)
        pipes.append(pipe)

    # Turn on depth streaming and color streaming for all devices
    StartStream(pipes)

    isExit = False
    while isExit == False:
        colorDatas = [None] * devCount
        depthDatas = [None] * devCount
        for index in range(devCount):
            if colorFrames[index] != None and depthFrames[index] != None:
                # Get the size and data of the frame
                colorSize = colorFrames[index].dataSize()
                colorData = np.asarray(colorFrames[index].data())
                colorWidth = colorFrames[index].width()
                colorHeight = colorFrames[index].height()
                colorFormat = colorFrames[index].format()
                depthSize = depthFrames[index].dataSize()
                depthData = np.asarray(depthFrames[index].data())
                depthWidth = depthFrames[index].width()
                depthHeight = depthFrames[index].height()
                valueScale = depthFrames[index].getValueScale();
                pipe = pipes[index]
                if colorData is not None and depthData is not None:
                    # Converting color frame data to OpenCV's Mat format
                    colorMat = colorData
                    if colorFormat == OB_PY_FORMAT_MJPG:
                        # Decode data frame MJPG to RGB format
                        colorMat = cv2.imdecode(colorMat,1)
                        # Resize the frame data to (height,width,3)
                        if colorMat is not None:
                            colorMat = np.resize(colorMat,(colorHeight, colorWidth, 3))
                    elif colorFormat == OB_PY_FORMAT_RGB888:
                        # Resize the frame data to (height,width,3)
                        colorMat = np.resize(colorMat,(colorHeight, colorWidth, 3))
                        # Convert frame data RGB to BGR
                        colorMat = cv2.cvtColor(colorMat, cv2.COLOR_RGB2BGR)
                    elif colorFormat == OB_PY_FORMAT_YUYV:
                        colorMat = np.resize(colorMat,(colorHeight, colorWidth, 2))
                        # Converting data frame YUYV to RGB data
                        colorMat = cv2.cvtColor(colorMat, cv2.COLOR_YUV2BGR_YUYV)
                    elif colorFormat == OB_PY_FORMAT_UYVY:
                        colorMat = np.resize(colorMat,(colorHeight, colorWidth, 2))
                        # Converting data frame YUYV to RGB data
                        colorMat = cv2.cvtColor(colorMat, cv2.COLOR_YUV2BGR_UYVY)
                    elif colorFormat == OB_PY_FORMAT_I420:
                        colorMat = colorMat.reshape((colorHeight * 3 // 2, colorWidth))
                        colorMat = cv2.cvtColor(colorMat, cv2.COLOR_YUV2BGR_I420)
                        colorMat = cv2.resize(colorMat, (colorWidth, colorHeight))
                        
                    # Convert depth frame data to OpenCV's Mat format
                    depthMat = depthData.reshape((depthHeight, depthWidth, 2))

                    # Convert frame data from 8bit to 16bit
                    depthMat = depthMat[:,:,0]+depthMat[:,:,1]*256          
                    # Conversion of frame data to 1mm units
                    depthMat = (depthMat * valueScale).astype('uint16')
                    # Rendering display
                    depthMat = (depthMat / 32).astype('uint8')
                    
                    # Convert frame data GRAY to RGB
                    depthMat = cv2.cvtColor(depthMat, cv2.COLOR_GRAY2RGB) 
                    
                    depthMat = cv2.resize(depthMat, (colorWidth, colorHeight))
                    
                    # Stitching color frames and depth frames together to reduce the depth image
                    depthMat = cv2.resize(depthMat, (depthWidth // 2, depthHeight // 2))

                    # Resize the color frame to the same size as the depth frame
                    if colorMat is not None:
                        colorMat = cv2.resize(colorMat, (depthWidth // 2, depthHeight // 2))
                        colorDatas[index] = colorMat
                        
                    depthDatas[index] = depthMat
                    
                    if colorDatas[index] is not None and depthDatas[index] is not None:
                        display = np.vstack((colorDatas[index],depthDatas[index]))
                        # Display the stitched image in the window
                        windowsName = str(pipe.getDevice().getDeviceInfo().name(),'utf-8') + " " + str(pipe.getDevice().getDeviceInfo().serialNumber(),'utf-8') + " color and depth"
                        cv2.imshow(windowsName, display)

                    key = cv2.waitKey(1)
                    # Press ESC or 'q' to close the window
                    if key == ESC or key == q:
                        isExit = True
                        break

    cv2.destroyAllWindows()

    for pipe in pipes:
        pipe.stop()

except ObException as e:
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
    e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
