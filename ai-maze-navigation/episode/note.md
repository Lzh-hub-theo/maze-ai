## Windows环境下将pip下载的第三方库下载到其他盘符下
1. 创建配置文件到指定目录： c://users/user_name/AppData/Roaming/pip/pip.ini
2. 编辑pip.ini
```ini
[global]
# pip默认安装路径
target = 存储第三方库文件夹的路径

# 国内镜像源
index-url = https://mirrors.aliyun.com/pypi/simple/
```
3. 配置环境变量（指定名称：**PYTHONPATH**），不用添加到Path中，只需设置环境变量并将存储第三方库文件夹的路径赋值给这个变量
4. 作用：让 pip show 库 找得到指定路径下的第三方库