from ObTypes import *
from Property import *
import Context
import Pipeline
import StreamProfile
import Device
import Filter
from Error import ObException
import cv2
import numpy as np
import os

import sys
import platform

plat = platform.system().lower()
if plat == 'windows':
    import msvcrt
elif plat == 'linux':
    import termios, tty

ESC = 27
r = 114
R = 82
d = 100
D = 68

# Save point cloud data to ply
def savePointsToPly(frame, depth_scale, fileName):
    directory = "./Points"
    if not os.path.exists(directory):
        os.makedirs(directory)

    fullPath = os.path.join(directory, fileName)
    points = frame.getPointCloudData()
    pointsSize = len(points)
    print("pointsSize = %d" % (pointsSize))
    with open(fullPath, "w") as fo:
        msg = "ply\nformat ascii 1.0\nelement vertex " + str(pointsSize) + "\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        fo.write(msg)
        for i in points:
            msg = f"{i.get('x') * depth_scale} {i.get('y') * depth_scale} {i.get('z') * depth_scale}\n"
            fo.write(msg)

# Save color point cloud data to ply
def saveRGBPointsToPly(frame, depth_scale, fileName):
    directory = "./RGBPoints"
    if not os.path.exists(directory):
        os.makedirs(directory)

    fullPath = os.path.join(directory, fileName)
    points = frame.getPointCloudData()
    pointsSize = len(points)
    print("pointsSize = %d" % (pointsSize))
    with open(fullPath, "w") as fo:
        msg = "ply\nformat ascii 1.0\nelement vertex " + str(pointsSize) + "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        fo.write(msg)
        for i in points:
            msg = f"{i.get('x') * depth_scale} {i.get('y') * depth_scale} {i.get('z') * depth_scale} {int(i.get('r'))} {int(i.get('g'))} {int(i.get('b'))}\n"
            fo.write(msg)

def getRGBPoints(pipeline, pointCloud):
    count = 0
    # Limit to a maximum of 10 repetitions
    while count < 10:
        count += 1
        # Wait for a frame of data with a timeout of 100ms
        frameset = pipeline.waitForFrames(100)
        if frameset is not None and frameset.depthFrame() is not None and frameset.colorFrame() is not None:
            try:
                # Generate and save color point clouds
                print("Save RGBD PointCloud ply file...")
                depth_frame = frameset.depthFrame()
                depth_scale = depth_frame.getValueScale()
                pointCloud.setCreatePointFormat(OB_PY_FORMAT_RGB_POINT)
                frame = pointCloud.process(frameset)
                saveRGBPointsToPly(frame, depth_scale, "RGBPoints.ply")
                print("RGBPoints.ply Saved")
            except ObException as e:
                print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
                    e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
            break

def getPoints(pipeline, pointCloud):
    count = 0
    # Limit to a maximum of 10 repetitions
    while count < 10:
        count += 1
        # Wait for a frame of data with a timeout of 100ms
        frameset = pipeline.waitForFrames(100)
        if frameset is not None and frameset.depthFrame() is not None:
            try:
                # Generate and save point clouds
                print("Save Depth PointCloud to ply file...")
                depth_frame = frameset.depthFrame()
                depth_scale = depth_frame.getValueScale()
                pointCloud.setCreatePointFormat(OB_PY_FORMAT_POINT)
                frame = pointCloud.process(frameset)
                savePointsToPly(frame, depth_scale, "DepthPoints.ply")
                print("DepthPoints.ply Saved")
            except ObException as e:
                print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
                    e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
            break

try:
    ctx = Context.Context(None)
    ctx.setLoggerSeverity(OB_PY_LOG_SEVERITY_ERROR)
    # Create pipeline
    pipeline = Pipeline.Pipeline(None, None)
    # Configure which streams to enable or disable in Pipeline by creating a Config
    config = Pipeline.Config()
    try:
        # Get all stream configurations for the depth camera, including the stream resolution, frame rate, and frame format
        profiles = pipeline.getStreamProfileList(OB_PY_SENSOR_DEPTH)

        videoProfile = None
        try:
            # Select the default resolution to open the stream,
            # you can configure the default resolution through the configuration file
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

    try:
        # Get all stream configurations for the color camera, including the stream resolution, frame rate, and frame format
        profiles = pipeline.getStreamProfileList(OB_PY_SENSOR_COLOR)

        videoProfile = None
        try:
            # Select the default resolution to open the stream,
            # you can configure the default resolution through the configuration file
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

    config.setAlignMode(OB_PY_ALIGN_D2C_HW_MODE)
    pipeline.start(config, None)

    # Create a point cloud Filter object (the creation of a point cloud Filter will get
    # the device parameters inside the Pipeline, so try to configure the device before the creation of the Filter)
    pointCloud = Filter.PointCloudFilter()

    cameraParam = pipeline.getCameraParam()
    pointCloud.setCameraParam(cameraParam)

    # Operation Tips
    print("Press R or r to create RGBD PointCloud and save to ply file! ")
    print("Press D or d to create Depth PointCloud and save to ply file! ")
    print("Press ESC to exit! ")

    while True:
        frameset = pipeline.waitForFrames(100)
        if plat == 'windows':
            if msvcrt.kbhit():
                key = ord(msvcrt.getch())
                # Press ESC key to exit
                if key == ESC:
                    print("stop")
                    break
                elif key == r or key == R:
                    getRGBPoints(pipeline, pointCloud)
                elif key == d or key == D:
                    getPoints(pipeline, pointCloud)

        elif plat == 'linux':
            assert sys.stdin.isatty(), "Can't run without a console to run on"
            fd = sys.stdin.fileno()
            stash = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                newterm = termios.tcgetattr(fd)
                newterm[tty.LFLAG] |= termios.ECHO
                termios.tcsetattr(fd, termios.TCSADRAIN, newterm)
                message = b''
                while True:
                    key = sys.stdin.buffer.read(1)
                    if key == b'\x1b':
                        print("stop")
                        pipeline.stop()
                        exit(0)
                    elif key == b'r' or key == b'R':
                        getRGBPoints(pipeline, pointCloud)
                    elif key == b'd' or key == b'D':
                        getPoints(pipeline, pointCloud)
                    elif key == b'\n':
                        break
                    else:
                        message += key
                message = message.decode(sys.stdin.encoding)
            finally:
                termios.tcsetattr(fd, termios.TCSANOW, stash)

    pipeline.stop()

except ObException as e:
    print("function: %s\nargs: %s\nmessage: %s\ntype: %d\nstatus: %d" % (
        e.getName(), e.getArgs(), e.getMessage(), e.getExceptionType(), e.getStatus()))
