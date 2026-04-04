# operation.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
import time

class Operation:
    def __init__(self, driver): 
        self.driver = driver
        self.actions = ActionChains(self.driver)

    # ===== 上移 =====
    def move_up(self, operation_time):
        self.actions.key_up('a')
        self.actions.key_up('d')
        self.actions.key_up('w')
        self.actions.key_up('s')
        self.actions.key_down('w')
        self.actions.perform()
        time.sleep(operation_time)

    # ===== 下移 =====
    def move_down(self, operation_time):
        self.actions.key_up('a')
        self.actions.key_up('d')
        self.actions.key_up('w')
        self.actions.key_up('s')
        self.actions.key_down('s')
        self.actions.perform() 
        time.sleep(operation_time)

    # ===== 左移 =====
    def move_left(self, operation_time):
        self.actions.key_up('a')
        self.actions.key_up('d')
        self.actions.key_up('w')
        self.actions.key_up('s')
        self.actions.key_down('a')
        self.actions.perform() 
        time.sleep(operation_time)

    # ===== 右移 =====
    def move_right(self, operation_time):
        self.actions.key_up('a')
        self.actions.key_up('d')
        self.actions.key_up('w')
        self.actions.key_up('s')
        self.actions.key_down('d')
        self.actions.perform()
        time.sleep(operation_time)

    # ===== 停止 =====
    def stop(self, operation_time):
        self.actions.key_up('a')
        self.actions.key_up('d')
        self.actions.key_up('w')
        self.actions.key_up('s')
        self.actions.perform()
        time.sleep(operation_time)

    # ===== 攻击 =====
    def shoot(self):
        self.actions.key_down('j')
        self.actions.perform()
        time.sleep(0.1)
        self.actions.key_up('j')
        self.actions.perform()

# # ===== 5. 测试控制 =====
# if __name__ == "__main__":
#     operation = Operation()
#     time.sleep(1.5)
#     operation.move_up(0.5)
#     operation.move_right(0.5)
#     operation.move_down(0.5)
#     operation.move_left(0.5)
#     operation.stop(0.5)
#     operation.shoot()
#     time.sleep(1)
#     operation.shoot()
#     time.sleep(1)
#     print("操作完成")