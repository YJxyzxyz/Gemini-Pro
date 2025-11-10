"""
虚拟画板 - 使用食指在空中绘画
功能：
- 食指伸出：绘画模式
- 竖起大拇指：擦除模式
- 握拳：清空画布
- 比划"V"手势：保存图片
"""

import cv2
import numpy as np
import mediapipe as mp
from ObTypes import *
import Pipeline
import Context
from Error import ObException
import time
from collections import deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
mp_drawing = mp.solutions.drawing_utils


class VirtualDrawingBoard:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.drawing_points = deque(maxlen=512)
        self.current_color = (255, 0, 0)  # 蓝色
        self.brush_size = 5
        self.eraser_size = 30
        self.mode = "draw"  # draw, erase
        
    def add_point(self, x, y):
        """添加绘画点"""
        point = (int(x * self.width), int(y * self.height))
        self.drawing_points.append(point)
        
    def draw_on_canvas(self):
        """在画布上绘制"""
        if len(self.drawing_points) > 1:
            for i in range(1, len(self.drawing_points)):
                if self.drawing_points[i - 1] is None or self.drawing_points[i] is None:
                    continue
                
                if self.mode == "draw":
                    cv2.line(
                        self.canvas,
                        self.drawing_points[i - 1],
                        self.drawing_points[i],
                        self.current_color,
                        self.brush_size
                    )
                elif self.mode == "erase":
                    cv2.circle(
                        self.canvas,
                        self.drawing_points[i],
                        self.eraser_size,
                        (0, 0, 0),
                        -1
                    )
    
    def clear_canvas(self):
        """清空画布"""
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.drawing_points.clear()
    
    def change_color(self):
        """切换颜色"""
        colors = [
            (255, 0, 0),    # 蓝色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 黄色
            (255, 255, 255) # 白色
        ]
        current_index = colors.index(self.current_color)
        self.current_color = colors[(current_index + 1) % len(colors)]
    
    def save_image(self, filename="drawing.png"):
        """保存图像"""
        cv2.imwrite(filename, self.canvas)
        print(f"图像已保存: {filename}")


def detect_gesture_for_drawing(hand_landmarks):
    """检测绘画手势"""
    if not hand_landmarks:
        return "No Hand", None
    
    landmarks = hand_landmarks.landmark
    
    # 获取食指尖位置
    index_tip = landmarks[8]
    finger_pos = (index_tip.x, index_tip.y)
    
    # 手指状态
    tip_ids = [4, 8, 12, 16, 20]
    finger_status = []
    
    # 拇指
    if landmarks[4].x < landmarks[3].x:
        finger_status.append(1)
    else:
        finger_status.append(0)
    
    # 其他手指
    for id in range(1, 5):
        if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y:
            finger_status.append(1)
        else:
            finger_status.append(0)
    
    # 手势判断
    if finger_status == [1, 0, 0, 0, 0]:  # 仅拇指
        return "Erase", finger_pos
    elif finger_status == [0, 1, 0, 0, 0]:  # 仅食指
        return "Draw", finger_pos
    elif finger_status == [0, 1, 1, 0, 0]:  # 食指+中指 (V手势)
        return "Save", finger_pos
    elif sum(finger_status) == 0:  # 握拳
        return "Clear", finger_pos
    elif sum(finger_status) == 5:  # 张开手掌
        return "ChangeColor", finger_pos
    
    return "None", finger_pos

def main():
    pipeline = None
    
    try:
        # 初始化SDK
        ctx = Context.Context(None)
        ctx.setLoggerSeverity(OB_PY_LOG_SEVERITY_ERROR)
        pipeline = Pipeline.Pipeline(None, None)
        config = Pipeline.Config()
        
        color_profiles = pipeline.getStreamProfileList(OB_PY_SENSOR_COLOR)
        color_profile = color_profiles.getVideoStreamProfile(640, 480, OB_PY_FORMAT_BGR, 30)
        config.enableStream(color_profile)
        
        pipeline.start(config, None)
        
        # 初始化画板
        board = VirtualDrawingBoard(640, 480)
        
        print("=== 虚拟画板控制说明 ===")
        print("1. 仅伸出食指：绘画模式")
        print("2. 仅竖起大拇指：擦除模式")
        print("3. 握拳：清空画布")
        print("4. 比划V手势：保存图片")
        print("5. 张开手掌：切换颜色")
        print("6. 按 'Q' 键退出")
        print("========================")
        
        last_gesture = ""
        gesture_start_time = time.time()
        is_drawing = False
        
        while True:
            frameset = pipeline.waitForFrames(100)
            if frameset is None:
                continue
            
            color_frame = frameset.colorFrame()
            if color_frame is None:
                continue
            
            frame = np.asanyarray(color_frame.get_data())
            frame = cv2.flip(frame, 1)  # 水平翻转，更直观
            
            # 处理手势
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            gesture = "No Hand"
            finger_pos = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    gesture, finger_pos = detect_gesture_for_drawing(hand_landmarks)
            
            # 手势控制
            current_time = time.time()
            
            if gesture == "Draw" and finger_pos:
                board.mode = "draw"
                board.add_point(finger_pos[0], finger_pos[1])
                is_drawing = True
            elif gesture == "Erase" and finger_pos:
                board.mode = "erase"
                board.add_point(finger_pos[0], finger_pos[1])
                is_drawing = True
            elif gesture == "Clear":
                if gesture != last_gesture:
                    board.clear_canvas()
                is_drawing = False
            elif gesture == "Save":
                if gesture != last_gesture:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    board.save_image(f"drawing_{timestamp}.png")
                is_drawing = False
            elif gesture == "ChangeColor":
                if gesture != last_gesture:
                    board.change_color()
                is_drawing = False
            else:
                if is_drawing:
                    board.drawing_points.append(None)  # 断开线条
                is_drawing = False
            
            last_gesture = gesture
            
            # 绘制到画布
            if is_drawing:
                board.draw_on_canvas()
            
            # 合成显示
            alpha = 0.6
            combined = cv2.addWeighted(frame, 1 - alpha, board.canvas, alpha, 0)
            
            # 显示当前模式和颜色
            mode_text = f"Mode: {board.mode.upper()} | Color: {board.current_color}"
            cv2.putText(combined, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, f"Gesture: {gesture}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示颜色指示器
            cv2.circle(combined, (600, 30), 20, board.current_color, -1)
            
            cv2.imshow("Virtual Drawing Board", combined)
            
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