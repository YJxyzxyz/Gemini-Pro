"""
增强版贪吃蛇游戏 - 使用Gemini Pro结构光相机和手势控制
特点：
1. 使用拇指尖作为蛇头控制
2. 竖起大拇指暂停/继续游戏
3. 握拳重新开始游戏
4. 实时显示游戏状态和得分
5. 碰撞检测和游戏结束逻辑
"""

import cv2
import numpy as np
import mediapipe as mp
from ObTypes import *
import Pipeline
import Context
from Error import ObException
import time
import random
from collections import deque

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
mp_drawing = mp.solutions.drawing_utils

# 游戏常量
GRID_SIZE = 20  # 网格大小
SNAKE_SPEED = 5  # 蛇的移动速度（帧数）
FOOD_COLOR = (0, 255, 0)  # 食物颜色（绿色）
SNAKE_COLOR = (255, 0, 0)  # 蛇身颜色（蓝色）
HEAD_COLOR = (0, 0, 255)  # 蛇头颜色（红色）
GRID_COLOR = (50, 50, 50)  # 网格颜色


class SnakeGame:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid_width = width // GRID_SIZE
        self.grid_height = height // GRID_SIZE
        self.reset()
        
    def reset(self):
        """重置游戏"""
        # 蛇的初始位置（中心）
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        self.snake = deque([(center_x, center_y)])
        self.direction = (1, 0)  # 初始方向：向右
        self.food = self.generate_food()
        self.score = 0
        self.game_over = False
        self.paused = False
        self.frame_count = 0
        
    def generate_food(self):
        """生成食物位置"""
        while True:
            food = (
                random.randint(0, self.grid_width - 1),
                random.randint(0, self.grid_height - 1)
            )
            if food not in self.snake:
                return food
    
    def set_direction_from_position(self, x, y):
        """根据手指位置设置蛇的移动方向"""
        if self.paused or self.game_over:
            return
            
        head = self.snake[0]
        grid_x = int(x * self.grid_width)
        grid_y = int(y * self.grid_height)
        
        # 计算方向向量
        dx = grid_x - head[0]
        dy = grid_y - head[1]
        
        # 归一化方向
        if abs(dx) > abs(dy):
            new_direction = (1 if dx > 0 else -1, 0)
        else:
            new_direction = (0, 1 if dy > 0 else -1)
        
        # 防止蛇反向移动
        if len(self.snake) > 1:
            opposite = (-self.direction[0], -self.direction[1])
            if new_direction != opposite:
                self.direction = new_direction
        else:
            self.direction = new_direction
    
    def update(self):
        """更新游戏状态"""
        if self.paused or self.game_over:
            return
        
        self.frame_count += 1
        if self.frame_count < SNAKE_SPEED:
            return
        
        self.frame_count = 0
        
        # 计算新的蛇头位置
        head = self.snake[0]
        new_head = (
            (head[0] + self.direction[0]) % self.grid_width,
            (head[1] + self.direction[1]) % self.grid_height
        )
        
        # 检查是否撞到自己
        if new_head in self.snake:
            self.game_over = True
            return
        
        # 移动蛇
        self.snake.appendleft(new_head)
        
        # 检查是否吃到食物
        if new_head == self.food:
            self.score += 10
            self.food = self.generate_food()
        else:
            self.snake.pop()
    
    def draw(self, frame):
        """绘制游戏画面"""
        # 绘制网格
        for x in range(0, self.width, GRID_SIZE):
            cv2.line(frame, (x, 0), (x, self.height), GRID_COLOR, 1)
        for y in range(0, self.height, GRID_SIZE):
            cv2.line(frame, (0, y), (self.width, y), GRID_COLOR, 1)
        
        # 绘制食物
        food_x = self.food[0] * GRID_SIZE
        food_y = self.food[1] * GRID_SIZE
        cv2.rectangle(
            frame,
            (food_x, food_y),
            (food_x + GRID_SIZE, food_y + GRID_SIZE),
            FOOD_COLOR,
            -1
        )
        
        # 绘制蛇身
        for i, segment in enumerate(self.snake):
            x = segment[0] * GRID_SIZE
            y = segment[1] * GRID_SIZE
            color = HEAD_COLOR if i == 0 else SNAKE_COLOR
            cv2.rectangle(
                frame,
                (x, y),
                (x + GRID_SIZE, y + GRID_SIZE),
                color,
                -1
            )
        
        # 绘制得分
        cv2.putText(
            frame,
            f"Score: {self.score}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        # 绘制游戏状态
        if self.game_over:
            self.draw_text_center(frame, "GAME OVER!", (0, 0, 255))
            self.draw_text_center(frame, "Fist to Restart", (255, 255, 255), 50)
        elif self.paused:
            self.draw_text_center(frame, "PAUSED", (255, 255, 0))
    
    def draw_text_center(self, frame, text, color, offset_y=0):
        """在屏幕中央绘制文本"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = (self.width - text_size[0]) // 2
        y = (self.height + text_size[1]) // 2 + offset_y
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def recognize_gesture(hand_landmarks):
    """识别手势"""
    if not hand_landmarks:
        return "No Hand", None
    
    landmarks = hand_landmarks.landmark
    
    # 获取拇指尖位置（用于控制）
    thumb_tip = landmarks[4]
    thumb_pos = (thumb_tip.x, thumb_tip.y)
    
    # 手指状态检测
    finger_status = []
    tip_ids = [4, 8, 12, 16, 20]
    
    # 拇指检测
    thumb_mcp = landmarks[2]
    if (landmarks[0].x < thumb_tip.x and thumb_tip.x < thumb_mcp.x) or 
            (landmarks[0].x > thumb_tip.x and thumb_tip.x > thumb_mcp.x):
        finger_status.append(1)
    else:
        finger_status.append(0)
    
    # 其他四根手指检测
    for id in range(1, 5):
        if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y:
            finger_status.append(1)
        else:
            finger_status.append(0)
    
    total_fingers = finger_status.count(1)
    
    # 手势判断
    if finger_status[0] == 1 and all(s == 0 for s in finger_status[1:]):
        return "Thumbs Up", thumb_pos
    elif total_fingers == 0:
        return "Fist", thumb_pos
    elif total_fingers >= 1:
        return "Control", thumb_pos
    
    return "Unknown", thumb_pos


def main():
    game = None
    pipeline = None
    
    try:
        # 初始化Orbbec SDK
        ctx = Context.Context(None)
        ctx.setLoggerSeverity(OB_PY_LOG_SEVERITY_ERROR)
        pipeline = Pipeline.Pipeline(None, None)
        config = Pipeline.Config()
        
        # 启用彩色数据流
        try:
            color_profiles = pipeline.getStreamProfileList(OB_PY_SENSOR_COLOR)
            color_profile = color_profiles.getVideoStreamProfile(640, 480, OB_PY_FORMAT_BGR, 30)
            if color_profile is None:
                print("未找到匹配的彩色流配置")
                return
            config.enableStream(color_profile)
        except ObException as e:
            print(f"无法启用彩色流: {e}")
            return
        
        pipeline.start(config, None)
        
        # 初始化游戏
        game = SnakeGame(640, 480)
        
        print("=== 贪吃蛇游戏控制说明 ===")
        print("1. 移动手掌，用拇指控制蛇的方向")
        print("2. 竖起大拇指：暂停/继续游戏")
        print("3. 握拳：重新开始游戏")
        print("4. 按 'Q' 键退出")
        print("========================")
        
        last_gesture = ""
        gesture_start_time = time.time()
        
        while True:
            frameset = pipeline.waitForFrames(100)
            if frameset is None:
                continue
            
            color_frame = frameset.colorFrame()
            if color_frame is None:
                continue
            
            # 获取图像数据
            frame = np.asanyarray(color_frame.get_data())
            
            # 创建游戏背景
            game_frame = np.zeros_like(frame)
            
            # MediaPipe处理
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            gesture = "No Hand"
            thumb_pos = None
            
            # 检测手势
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 在游戏画面上绘制手部关键点
                    mp_drawing.draw_landmarks(
                        game_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                    )
                    
                    gesture, thumb_pos = recognize_gesture(hand_landmarks)
            
            # 手势控制逻辑
            current_time = time.time()
            if gesture != last_gesture:
                gesture_start_time = current_time
                last_gesture = gesture
            
            gesture_duration = current_time - gesture_start_time
            
            # 竖起大拇指持续0.5秒：暂停/继续
            if gesture == "Thumbs Up" and gesture_duration > 0.5:
                game.paused = not game.paused
                last_gesture = ""
            
            # 握拳持续0.5秒：重新开始
            elif gesture == "Fist" and gesture_duration > 0.5 and game.game_over:
                game.reset()
                last_gesture = ""
            
            # 控制蛇的方向
            elif gesture == "Control" and thumb_pos:
                game.set_direction_from_position(thumb_pos[0], thumb_pos[1])
            
            # 更新游戏
            game.update()
            
            # 绘制游戏
            game.draw(game_frame)
            
            # 显示手势状态
            status_text = f"Gesture: {gesture}"
            if game.paused:
                status_text += " | PAUSED"
            elif game.game_over:
                status_text += " | GAME OVER"
            
            cv2.putText(
                game_frame,
                status_text,
                (10, game_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # 显示画面
            cv2.imshow("Snake Game with Gesture Control", game_frame)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except ObException as e:
        print(f"SDK 异常: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        if pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
