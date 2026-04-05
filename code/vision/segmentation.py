import cv2
import numpy as np
from object.enemy import EnemyDetector

class Segmenter:
    def __init__(self):
        self.kernel = np.ones((3, 3), np.uint8)

    def get_masks(self, img):
        """
        输入：BGR图像
        输出：所有mask
        """

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        masks = {}

        # 玩家（黄色）
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        masks["player"] = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 敌人（红色）
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + \
                   cv2.inRange(hsv, lower_red2, upper_red2)

        masks["enemy_red"] = mask_red

        # 敌人（白色）
        # --- 1. 定义深青色 (车身主体) ---
        # H: ~95, S: 中高, V: 中低
        lower_cyan = np.array([85, 80, 30])
        upper_cyan = np.array([105, 255, 100])
        mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)

        # --- 2. 定义浅灰色 (履带部分) ---
        # S: 低 (0-60), V: 中等 (130-200)
        # 注意：灰色的 H 可以设为 0-180，因为我们只关心 S 和 V
        lower_gray = np.array([0, 0, 130])
        upper_gray = np.array([180, 60, 200])
        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

        # --- 3. 定义白色 (炮管高光/细节) ---
        # S: 极低, V: 极高
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # --- 4. 合并所有掩膜 ---
        # 使用按位或运算，将三个掩膜合二为一
        mask_white_tank = cv2.bitwise_or(mask_cyan, mask_gray)
        mask_white_tank = cv2.bitwise_or(mask_white_tank, mask_white)

        # --- 5. 后处理 (可选但推荐) ---
        # 去除噪点，让检测更稳定
        kernel = np.ones((3,3), np.uint8)
        mask_white_tank = cv2.morphologyEx(mask_white_tank, cv2.MORPH_OPEN, kernel) # 开运算去噪
        mask_white_tank = cv2.morphologyEx(mask_white_tank, cv2.MORPH_CLOSE, kernel) # 闭运算填充空洞

        masks["enemy_white"] = mask_white_tank

        # 砖墙（棕色）
        lower_brick = np.array([10, 100, 100])
        upper_brick = np.array([25, 255, 200])
        masks["brick"] = cv2.inRange(hsv, lower_brick, upper_brick)

        # 钢墙（灰色）
        lower_steel = np.array([0, 0, 80])
        upper_steel = np.array([180, 40, 200])
        masks["steel"] = cv2.inRange(hsv, lower_steel, upper_steel)

        # 水（蓝色）
        lower_water = np.array([90, 100, 100])
        upper_water = np.array([130, 255, 255])
        masks["water"] = cv2.inRange(hsv, lower_water, upper_water)

        # 草（绿色）
        lower_grass = np.array([35, 80, 80])
        upper_grass = np.array([85, 255, 255])
        masks["grass"] = cv2.inRange(hsv, lower_grass, upper_grass)

        # 子弹（白色小点）
        masks["bullet"] = mask_white.copy()

        # buff（高亮）
        lower_buff = np.array([0, 150, 200])
        upper_buff = np.array([180, 255, 255])
        masks["buff"] = cv2.inRange(hsv, lower_buff, upper_buff)

        # 统一做形态学处理
        # for key in ["enemy_red", "player", "enemy_white"]:
        for key in masks:
            masks[key] = cv2.dilate(masks[key], self.kernel, iterations=1)
            masks[key] = cv2.erode(masks[key], self.kernel, iterations=1)

        return masks