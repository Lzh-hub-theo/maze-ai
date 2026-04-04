#opengame.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
import time

class OpenGame:
    def __init__(self):
        # ===== 1 配置 Chrome 选项  =====
        chrome_options = Options()

        chrome_options.add_argument("--no-sandbox")          # 解决权限/沙箱问题
        chrome_options.add_argument("--disable-dev-shm-usage") # 解决共享内存不足
        chrome_options.add_argument("--disable-gpu")         # 禁用 GPU 加速
        chrome_options.add_argument("--window-size=1280,720")  # 指定窗口大小

        # ===== 2 启动浏览器 =====
        try:
            self.driver = webdriver.Chrome(options=chrome_options) 
            print("浏览器启动成功！")
        except Exception as e:
            print("启动失败:", e)
            input("按回车退出...")
            exit()

        self.driver.get("https://battle-city.js.org/#/")

        # ===== 3. 等待页面加载 =====
        time.sleep(2)

        # ===== 4. 点击页面以获取焦点（非常关键！）=====
        body = self.driver.find_element(By.TAG_NAME, "body")
        body.click()
        body.send_keys("j") # 进入单机模式
        time.sleep(0.5)
        body.send_keys("j") # 进入第一关

        time.sleep(1)
