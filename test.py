import urllib.parse
import json
import paramiko
import stat
import sys
import os
import time
import importlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import plots

# Establish SSH connection
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('169.254.92.202', username='analog', password='analog')

# Create SFTP client
sftp = ssh.open_sftp()

remote_file_path = '/home/analog/Desktop/code/dict.json'


# print(sftp.open(remote_file_path).__dir__())
# print(sftp.open(remote_file_path).read())
data_json = sftp.open(remote_file_path).read()

data = json.loads(data_json)
# print(data.keys())

thetaScan = data["thetaScan"]
results = data["results"]

plots.plot_polar(thetaScan, results, title="MuSiC")

# sftp.chmod(remote_file_path, mode=stat.S_IRWXO )

# remote_file = sftp.open('/home/analog/Desktop/code/dict.json')

# print(remote_file)

# if sftp is not None:
#     ssh.close()

# # Read and parse JSON data
# with open(local_file_path, 'r') as f:
#     json_data = json.load(f)

# # Save JSON data to a local variable
# my_json_data = json_data

# # Close the SSH connection
# sftp.close()
# ssh.close()

# # Now you can use the 'my_json_data' variable in your Python code
# print(my_json_data)


# importlib.reload(plots)

# while(True):
#     os.system(f'sshpass -p analog ssh analog@169.254.92.202 ' 
#               + '"cd /home/analog/Desktop/code; "' 
#               + 'python3 import json; ')
    
#     time.sleep(0.5)
    



# dict = json.loads(urllib.parse.unquote(sys.argv[1])) 

# thetaScan = dict.thetaScan
# results = dict.results

# plots.plot_polar(thetaScan, results, title="MuSiC")
# plots.plot_regular(thetaScan, results, title="MuSiC")

# peaks, _ = signal.find_peaks(results, height=-2)
# doas = thetaScan[peaks]

# print("DoA:", doas * 180 / np.pi)