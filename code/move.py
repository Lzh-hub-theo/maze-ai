from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
import time

# ===== 1. 启动浏览器 =====
# ===== 1.1 配置 Chrome 选项  =====
chrome_options = Options()

# --- 必须的 Linux 参数 ---
chrome_options.add_argument("--no-sandbox")          # 解决权限/沙箱问题
chrome_options.add_argument("--disable-dev-shm-usage") # 解决共享内存不足
chrome_options.add_argument("--disable-gpu")         # 禁用 GPU 加速
chrome_options.add_argument("--window-size=1280,720")  # 指定窗口大小

# ===== 1.2 启动浏览器 =====
try:
    driver = webdriver.Chrome(options=chrome_options) 
    print("浏览器启动成功！")
except Exception as e:
    print("启动失败:", e)
    input("按回车退出...")
    exit()

driver.get("https://battle-city.js.org/#/")

# ===== 2. 等待页面加载 =====
time.sleep(3)

# ===== 3. 点击页面以获取焦点（非常关键！）=====
body = driver.find_element(By.TAG_NAME, "body")
body.click()
body.send_keys("j") # 进入单机模式
time.sleep(0.5)
body.send_keys("j") # 进入第一关

time.sleep(1)

# --- 进入游戏 ---
actions = ActionChains(driver)
# actions.send_keys(Keys.RETURN).perform() # 假设回车或 J 进游戏

# ===== 4. 定义动作函数 =====
def move_up(operation_time):
    actions.key_up('a')
    actions.key_up('d')
    actions.key_up('w')
    actions.key_up('s')
    actions.key_down('w')
    actions.perform()
    time.sleep(operation_time)

def move_down(operation_time):
    actions.key_up('a')
    actions.key_up('d')
    actions.key_up('w')
    actions.key_up('s')
    actions.key_down('s')
    actions.perform() 
    time.sleep(operation_time)

def move_left(operation_time):
    actions.key_up('a')
    actions.key_up('d')
    actions.key_up('w')
    actions.key_up('s')
    actions.key_down('a')
    actions.perform() 
    time.sleep(operation_time)

def move_right(operation_time):
    actions.key_up('a')
    actions.key_up('d')
    actions.key_up('w')
    actions.key_up('s')
    actions.key_down('d')
    actions.perform()
    time.sleep(operation_time)

def stop(operation_time):
    actions.key_up('a')
    actions.key_up('d')
    actions.key_up('w')
    actions.key_up('s')
    actions.perform()
    time.sleep(operation_time)

def shoot():
    actions.key_down('j')
    actions.perform()
    time.sleep(0.1)
    actions.key_up('j')
    actions.perform()
# ===== 5. 测试控制 =====

time.sleep(1.5)

move_up(0.5)

move_right(0.5)

move_down(0.5)

move_left(0.5)

stop(0.5)

shoot()

time.sleep(1)

shoot()

time.sleep(1)

print("操作完成")