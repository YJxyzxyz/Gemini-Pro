import cv2
import numpy as np
import mediapipe as mp
from ObTypes import *
import Pipeline
import Context
from Error import ObException

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 手指指尖的ID
tip_ids = [4, 8, 12, 16, 20]


def recognize_gesture(hand_landmarks):
    """
    根据手部关键点识别手势
    """
    if not hand_landmarks:
        return "No Hand"

    landmarks = hand_landmarks.landmark
    finger_status = []

    # --- 拇指检测 ---
    # 检查拇指是张开还是闭合 (基于拇指尖相对于手腕的x坐标)
    # 注意: 这个逻辑假设手是垂直的。更复杂的逻辑需要考虑手的旋转。
    thumb_tip = landmarks[tip_ids[0]]
    thumb_mcp = landmarks[tip_ids[0] - 2]  # 拇指的中间关节
    if (hand_landmarks.landmark[0].x < landmarks[tip_ids[0]].x and thumb_tip.x < thumb_mcp.x) or \
            (hand_landmarks.landmark[0].x > landmarks[tip_ids[0]].x and thumb_tip.x > thumb_mcp.x):
        finger_status.append(1)
    else:
        finger_status.append(0)

    # --- 其余四根手指检测 ---
    for id in range(1, 5):
        # 检查手指是伸出还是弯曲 (基于指尖和下面关节的y坐标)
        if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y:
            finger_status.append(1)
        else:
            finger_status.append(0)

    # --- 手势判断 ---
    total_fingers = finger_status.count(1)

    # 竖起大拇指手势
    if finger_status[0] == 1 and all(s == 0 for s in finger_status[1:]):
        return "Thumbs Up"

    return f"Fingers: {total_fingers}"


def main():
    try:
        # 初始化Orbbec SDK
        ctx = Context.Context(None)
        ctx.setLoggerSeverity(OB_PY_LOG_SEVERITY_ERROR)
        pipeline = Pipeline.Pipeline(None, None)
        config = Pipeline.Config()

        # 仅使能彩色数据流
        try:
            color_profiles = pipeline.getStreamProfileList(OB_PY_SENSOR_COLOR)
            # 选择一个合适的分辨率，640x480对于手势识别来说足够了
            color_profile = color_profiles.getVideoStreamProfile(640, 480, OB_PY_FORMAT_BGR, 30)
            if color_profile is None:
                print("未找到匹配的彩色流配置")
                return
            config.enableStream(color_profile)
        except ObException as e:
            print(f"无法使能彩色流: {e}")
            return

        pipeline.start(config, None)

        print("按 'Q' 键退出...")

        while True:
            frameset = pipeline.waitForFrames(100)
            if frameset is None:
                continue

            color_frame = frameset.colorFrame()
            if color_frame is None:
                continue

            # 获取图像数据
            frame = np.asanyarray(color_frame.get_data())

            # MediaPipe需要RGB格式，而我们从SDK得到的是BGR格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 为了提高性能，将图像标记为不可写
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            gesture = "No Hand"
            # 如果检测到手
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 在图像上绘制手部关键点和连接
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # 识别手势
                    gesture = recognize_gesture(hand_landmarks)

            # 在画面上显示识别出的手势
            cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 2, cv2.LINE_AA)

            # 显示结果
            cv2.imshow("Gesture Recognition", frame)

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except ObException as e:
        print(f"SDK 异常: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 清理资源
        if 'pipeline' in locals():
            pipeline.stop()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()