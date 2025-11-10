"""
3D物体交互系统
功能：
- 用手势抓取、移动、旋转虚拟3D物体
- 支持多种手势控制
- 实时深度感知
"""

import cv2
import numpy as np
import mediapipe as mp
from ObTypes import *
import Pipeline
import Context
from Error import ObException
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
)


class Virtual3DObject:
    def __init__(self, x, y, size=50):
        self.x = x
        self.y = y
        self.z = 0
        self.size = size
        self.rotation = 0
        self.color = (255, 100, 100)
        self.grabbed = False
        
    def draw(self, frame):
        """绘制3D立方体投影"""
        # 简化的立方体绘制
        points = []
        for i in range(-1, 2, 2):
            for j in range(-1, 2, 2):
                x = int(self.x + i * self.size * math.cos(self.rotation))
                y = int(self.y + j * self.size * math.sin(self.rotation))
                points.append((x, y))
        
        # 绘制立方体边框
        for i in range(0, len(points), 2):
            cv2.line(frame, points[i], points[(i + 1) % len(points)],
                    self.color, 2)
        
        # 显示抓取状态
        if self.grabbed:
            cv2.circle(frame, (int(self.x), int(self.y)), 10, (0, 255, 0), -1)
    
    def is_grabbed(self, hand_x, hand_y):
        """检测是否被抓取"""
        distance = math.sqrt((self.x - hand_x)**2 + (self.y - hand_y)**2)
        return distance < self.size
    
    def update_position(self, x, y):
        """更新位置"""
        self.x = x
        self.y = y


def detect_pinch_gesture(hand_landmarks):
    """检测捏合手势"""
    if not hand_landmarks:
        return False, None
    
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    
    distance = math.sqrt(
        (thumb_tip.x - index_tip.x)**2 +
        (thumb_tip.y - index_tip.y)**2
    )
    
    is_pinching = distance < 0.05
    position = ((thumb_tip.x + index_tip.x) / 2, (thumb_tip.y + index_tip.y) / 2)
    
    return is_pinching, position

def main():
    pipeline = None
    
    try:
        ctx = Context.Context(None)
        ctx.setLoggerSeverity(OB_PY_LOG_SEVERITY_ERROR)
        pipeline = Pipeline.Pipeline(None, None)
        config = Pipeline.Config()
        
        color_profiles = pipeline.getStreamProfileList(OB_PY_SENSOR_COLOR)
        color_profile = color_profiles.getVideoStreamProfile(640, 480, OB_PY_FORMAT_BGR, 30)
        config.enableStream(color_profile)
        
        pipeline.start(config, None)
        
        # 创建多个3D物体
        objects = [
            Virtual3DObject(200, 200, 40),
            Virtual3DObject(400, 300, 50),
            Virtual3DObject(500, 150, 45)
        ]
        
        grabbed_object = None
        
        print("=== 3D物体交互控制 ===")
        print("1. 捏合手势（拇指+食指）：抓取物体")
        print("2. 移动手：移动被抓取的物体")
        print("3. 松开捏合：释放物体")
        print("4. 按 'Q' 键退出")
        print("=====================")
        
        while True:
            frameset = pipeline.waitForFrames(100)
            if frameset is None:
                continue
            
            color_frame = frameset.colorFrame()
            if color_frame is None:
                continue
            
            frame = np.asanyarray(color_frame.get_data())
            frame = cv2.flip(frame, 1)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            is_pinching = False
            pinch_position = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    is_pinching, pinch_position = detect_pinch_gesture(hand_landmarks)
                    
                    if is_pinching and pinch_position:
                        pinch_x = int(pinch_position[0] * frame.shape[1])
                        pinch_y = int(pinch_position[1] * frame.shape[0])
                        
                        # 尝试抓取物体
                        if grabbed_object is None:
                            for obj in objects:
                                if obj.is_grabbed(pinch_x, pinch_y):
                                    grabbed_object = obj
                                    grabbed_object.grabbed = True
                                    break
                        
                        # 移动被抓取的物体
                        if grabbed_object:
                            grabbed_object.update_position(pinch_x, pinch_y)
                        
                        # 绘制捏合点
                        cv2.circle(frame, (pinch_x, pinch_y), 10, (0, 255, 255), -1)
            
            # 释放物体
            if not is_pinching and grabbed_object:
                grabbed_object.grabbed = False
                grabbed_object = None
            
            # 绘制所有物体
            for obj in objects:
                obj.draw(frame)
            
            cv2.imshow("Interactive 3D Objects", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()