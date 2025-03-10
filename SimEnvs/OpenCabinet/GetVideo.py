import os
from ftplib import FTP

# FTP服务器信息
ftp_server = 'ftp.example.com'
ftp_user = 'username'
ftp_pass = 'password'
remote_folder = '/path/to/remote/folder'
local_folder = 'downloaded_mp4_files'

# 创建本地保存目录
os.makedirs(local_folder, exist_ok=True)

def download_file(ftp, filename, local_path):
    with open(local_path, 'wb') as f:
        ftp.retrbinary(f'RETR {filename}', f.write)

# 连接到FTP服务器
ftp = FTP(ftp_server)
ftp.login(ftp_user, ftp_pass)

# 切换到目标目录
ftp.cwd(remote_folder)

# 获取目录列表
files = ftp.nlst()

# 过滤并下载mp4文件
for file in files:
    if file.endswith('.mp4'):
        local_file_path = os.path.join(local_folder, file)
        print(f'Downloading {file} to {local_file_path}')
        download_file(ftp, file, local_file_path)
        print(f'{file} downloaded successfully.')

# 退出FTP连接
ftp.quit()



