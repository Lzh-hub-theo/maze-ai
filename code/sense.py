# sense.py
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from PIL import Image
import cv2
import time
from opengame import OpenGame 
from operation import Operation 
from vision.segmentation import Segmenter
from vision.hsv_tuner import HSVTuner

class TankGameEnv:
    def __init__(self):
        self.debug = True
        self.game = OpenGame() 
        self.driver = self.game.driver
        self.operation = Operation(self.driver)
        self.segmenter = Segmenter()

        self.action_map = {
            0: self.operation.move_up,    # 上
            1: self.operation.move_down,  # 下
            2: self.operation.move_left,  # 左
            3: self.operation.move_right, # 右
            4: self.operation.shoot,      # 攻击 (开火)
            5: self.operation.stop        # 停止 (新增的显式停止动作)
        }

    """ 截图 """
    def capture_screen(self):
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
            
        return img, (width, height)

    """ 裁剪 """
    def crop_game_area(self, img):
        h, w, _ = img.shape
        y1 = int(32)
        y2 = int(449)
        x1 = int(32)
        x2 = int(449)
        return img[y1:y2, x1:x2]

    """ 掩膜+形状+过滤 -> 语义信息 """
    def detect_game_state(self, img):
        state = {
            'player_pos': None,
            'enemy_count': 0,
            'enemy_positions': []
        }

        self.segmenter.detect_objects(img, state)

        return state, img

    def step(self, action):
        """
        执行一个动作。
        这里的 action 是 self.action_map 的键值 (0-5)
        """
        # 获取对应的动作函数
        action_func = self.action_map.get(action)
        
        if action_func:
            if action in [0, 1, 2, 3]: # 移动类
                action_func(0.3) # 持续 0.3 秒
            elif action == 4: # 射击类
                action_func() # 调用 shoot
                time.sleep(0.2) # 射击后短暂冷却
            elif action == 5: # 停止类
                action_func(0.1)
        
    """ 主循环 """
    def run_demo(self):
        print("Demo 开始。按 Ctrl+C 或 关闭窗口 停止。")
        
        while True:
            # time.sleep(0.03)
            # time.sleep(5)
            try:
                raw_img, win_size = self.capture_screen()
                raw_img = self.crop_game_area(raw_img)
                state, annotated_img = self.detect_game_state(raw_img)
                if self.debug:
                    cv2.imshow("Cropped", annotated_img)
                    # tuner = HSVTuner() # HSV提取颜色
                    # tuner.run(raw_img)
                    # break
                
                # 打印状态信息
                print(f"玩家位置: {state['player_pos']}, 敌人数量: {state['enemy_count']}")

                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"运行出错: {e}")
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    env = TankGameEnv()
    env.run_demo()
