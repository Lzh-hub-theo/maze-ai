import cv2
import numpy as np
from .entity.enemy import EnemyDetector
from .entity.player import PlayerTankDetector
from .entity.wall import WallDetector
from .entity.ironwall import IronWallDetector
from .entity.river import RiverDetector
from .entity.grass import GrassDetector
from .entity.bullet import BulletDetector
from .entity.base import BaseDetector

class Segmenter:
    def __init__(self):
        self.kernel = np.ones((3, 3), np.uint8)

    def get_masks(self, img):
        # 输入：BGR图像，输出：所有mask
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        masks = {}

        masks["enemy"] = EnemyDetector().get_mask(hsv)
        masks["player"] = PlayerTankDetector().get_mask(hsv)
        masks["brick"] = WallDetector().get_mask(hsv)
        masks["steel"] = IronWallDetector().get_mask(hsv)
        masks["water"] = RiverDetector().get_mask(hsv)
        masks["grass"] = GrassDetector().get_mask(hsv)
        masks["bullet"] = BulletDetector().get_mask(hsv)
        masks["base"] = BaseDetector().get_mask(hsv)

        return masks