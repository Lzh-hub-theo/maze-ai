# sense.py
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from PIL import Image
import cv2
import time
import random
from opengame import OpenGame 
from operation import Operation 

class TankGameEnv:
    def __init__(self):
        self.game = OpenGame() 
        self.driver = self.game.driver
        self.operation = Operation(self.driver)

        self.action_map = {
            0: self.operation.move_up,    # 上
            1: self.operation.move_down,  # 下
            2: self.operation.move_left,  # 左
            3: self.operation.move_right, # 右
            4: self.operation.shoot,      # 攻击 (开火)
            5: self.operation.stop        # 停止 (新增的显式停止动作)
        }

    def capture_screen(self):
        """
        使用 Selenium 截图，并转换为 OpenCV 格式
        """
        # 获取当前窗口大小
        window_size = self.driver.get_window_size()
        width = window_size['width']
        height = window_size['height']
        
        # 执行截图
        screenshot = self.driver.get_screenshot_as_png() 
        
        # 使用 OpenCV 读取图片
        nparr = np.frombuffer(screenshot, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise Exception("截图读取失败")
            
        # print(f"截图成功，分辨率: w:{width}, h:{height}")
        return img, (width, height)

    def crop_game_area(self, img):
        """
        裁剪游戏区域（去掉UI、边框）
        需要你根据实际截图微调参数
        """
        h, w, _ = img.shape

        y1 = int(26)
        y2 = int(442)
        x1 = int(36)
        x2 = int(450)

        return img[y1:y2, x1:x2]

    def preprocess_image(self, img):
        """
        图像预处理：裁剪掉非游戏区域（如黑边），调整大小
        注意：根据你的显示器分辨率，可能需要调整裁剪参数。
        """
        # 转为灰度图或HSV用于颜色分割（可选）
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return gray

    def detect_game_state(self, img):
        """
        核心状态检测函数。
        逻辑：
        1. 转换颜色空间 (BGR -> RGB -> HSV) 以确保颜色阈值准确。
        2. 黄色掩膜 -> 检测玩家坦克。
        3. 红色 + 白色 掩膜 -> 检测敌方坦克。
        """
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        state = {
            'player_pos': None,
            'enemy_count': 0,
            'enemy_positions': []
        }

        # --- 1. 玩家坦克检测 (黄色) ---
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_player = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        kernel = np.ones((3,3), np.uint8)
        mask_player = cv2.dilate(mask_player, kernel, iterations=2)
        mask_player = cv2.erode(mask_player, kernel, iterations=1)

        contours_player, _ = cv2.findContours(mask_player, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours_player:
            area = cv2.contourArea(cnt)
            if 200 < area < 2000: 
                x, y, w, h = cv2.boundingRect(cnt)
                center = (x + w//2, y + h//2)
                state['player_pos'] = center
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, "Player", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- 2. 敌方坦克检测 ---
        # A. 红色
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_enemy_red = mask_red1 + mask_red2

        # B. 白色
        lower_white = np.array([0, 0, 200]) 
        upper_white = np.array([10, 20, 255])
        mask_enemy_white = cv2.inRange(hsv, lower_white, upper_white)

        # C. 合并
        mask_enemy = cv2.bitwise_or(mask_enemy_red, mask_enemy_white)
        mask_enemy = cv2.dilate(mask_enemy, kernel, iterations=2)
        mask_enemy = cv2.erode(mask_enemy, kernel, iterations=1)

        contours_enemy, _ = cv2.findContours(mask_enemy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours_enemy:
            area = cv2.contourArea(cnt)
            if 200 < area < 2000:
                x, y, w, h = cv2.boundingRect(cnt)
                center = (x + w//2, y + h//2)
                state['enemy_positions'].append(center)
                state['enemy_count'] += 1
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(img, "Enemy", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return state, img

    def step(self, action):
        """
        执行一个动作。
        这里的 action 是 self.action_map 的键值 (0-5)
        """
        # 获取对应的动作函数
        action_func = self.action_map.get(action)
        
        if action_func:
            # 注意：move 函数需要时间参数，shoot 不需要
            if action in [0, 1, 2, 3]: # 移动类
                action_func(0.3) # 持续 0.3 秒
            elif action == 4: # 射击类
                action_func() # 调用 shoot
                time.sleep(0.2) # 射击后短暂冷却
            elif action == 5: # 停止类
                action_func(0.1)
        
        # --- 可选：返回奖励和是否结束 ---
        # 这里为了简化，只做动作，不返回 RL 标准的 (obs, reward, done, info)
        # 如果需要做强化学习，这里需要补充逻辑

    def run_demo(self):
        """
        演示函数：自动运行并显示检测画面
        """
        print("Demo 开始。按 Ctrl+C 或 关闭窗口 停止。")
        
        while True:
            try:
                raw_img, win_size = self.capture_screen()
                raw_img = self.crop_game_area(raw_img)
                state, annotated_img = self.detect_game_state(raw_img)
                cv2.imshow("Cropped", raw_img)
                
                # # 打印状态信息
                # print(f"玩家位置: {state['player_pos']}, 敌人数量: {state['enemy_count']}")

                # # 随机选择一个动作
                # action = random.randint(0, 5) # 包含 0 到 5
                # self.step(action)
                
                # # 调整图像大小以便显示
                # display_img = cv2.resize(annotated_img, (600, 600))
                # cv2.imshow('Game View - Player (Green) vs Enemy (Red)', display_img)
                
                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"运行出错: {e}")
                break
        
        cv2.destroyAllWindows()
        # 注意：不要在这里 quit driver，除非确定结束
        # self.driver.quit()

if __name__ == "__main__":
    env = TankGameEnv()
    env.run_demo()