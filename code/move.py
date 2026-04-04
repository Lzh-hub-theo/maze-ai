from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# ===== 1. 启动浏览器 =====
# ===== 1.1 配置 Chrome 选项  =====
chrome_options = Options()

# --- 必须的 Linux 参数 ---
chrome_options.add_argument("--no-sandbox")          # 解决权限/沙箱问题
chrome_options.add_argument("--disable-dev-shm-usage") # 解决共享内存不足
chrome_options.add_argument("--disable-gpu")         # 禁用 GPU 加速
chrome_options.add_argument("--window-size=1280,720")  # 指定窗口大小
chrome_options.add_argument("--headless=new")        # 虚拟机没有GUI，加这一行

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

print("开始控制坦克...")

# ===== 4. 定义动作函数 =====
def move_up():
    body.send_keys("w")

def move_down():
    body.send_keys("s")

def move_left():
    body.send_keys("a")

def move_right():
    body.send_keys("d")

def shoot():
    body.send_keys("j")

# ===== 5. 测试控制 =====
time.sleep(2)

# 示例：移动 + 射击
move_up()
time.sleep(0.5)

move_right()
time.sleep(0.5)

shoot()
time.sleep(0.5)

move_left()
time.sleep(0.5)

shoot()

print("操作完成")