from tqdm import tqdm
from update import update, initialize
try :
    from config import MN5_exec_host, MN5_transfer_host, allLocalPaths, allRemotesPaths
except:
    print("Config file not found or wrongly written.\nSee tests/config_example.py for futher informations.")

    
    
from data.utils.utils import isdir_create_remote, isdir_create

print("Configuration will take a few minutes. The speed depends on your internet connection")

for localPath in tqdm(allLocalPaths):
    isdir_create(localPath)

for remotePath in tqdm(allRemotesPaths):
    isdir_create_remote(remotePath)


initialize()
