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

    def detect_objects(self, img, state):
        # 输入：BGR图像，输出：所有mask
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        EnemyDetector().detect_object(hsv, img, state)
        PlayerTankDetector().detect_object(hsv, img, state)
        WallDetector().detect_object(hsv, img)
        IronWallDetector().detect_object(hsv, img)
        RiverDetector().detect_object(hsv, img)
        GrassDetector().detect_object(hsv, img)
        BulletDetector().detect_object(hsv, img)
        BaseDetector().detect_object(hsv, img)
