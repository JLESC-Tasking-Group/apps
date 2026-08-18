import os
import subprocess
from typing import List
try :
    from config import MN5_exec_host, MN5_transfer_host, sbatch_queue, MN5_remote_folder
except:
    print("Config file not found or wrongly written.\nSee tests/config_example.py for futher informations.")
    

# Execute commands on local machine
def exec_local_cmds(cmdList :List[str], display :bool=False) -> subprocess.CompletedProcess[str] :
    if display:
        res = subprocess.run(cmdList, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(res.stdout)
        print(res.stderr)
        if res.returncode != 0:
            print(cmdList)
            raise Exception
    return subprocess.run(cmdList, capture_output=True, check=True, text=True)
    
# Execute commands on MN5
def exec_remote_cmds(cmdList : List[str], display :bool=False) -> subprocess.CompletedProcess[str] :
    cmdString = " && ".join(cmdList)
    if display:
        res = subprocess.run(["ssh", MN5_exec_host, cmdString], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(res.stdout)
        print(res.stderr)
        if res.returncode != 0:
            print(cmdList)
            raise Exception
        return res
    res =  subprocess.run(["ssh", MN5_exec_host, cmdString], capture_output=True, text=True)
    if res.returncode != 0:
        print(cmdList)
        print(res.stdout)
        print(res.stderr)
        raise Exception
    return res

# Transfer data from MN5 to local machine
def transfer_from_remote(destAbsPath :str, inpAbsPath :str, rec :bool=False) -> None:
    if rec :
        subprocess.run(["scp", "-r", f"{MN5_transfer_host}:{inpAbsPath}", f"{destAbsPath}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    else :
        subprocess.run(["scp", f"{MN5_transfer_host}:{inpAbsPath}", f"{destAbsPath}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

# Transfer data from local machine to MN5
def transfer_to_remote(destAbsPath :str, inpAbsPath :str, rec :bool=False) -> None:
    if rec :
        subprocess.run(["scp", "-r", f"{inpAbsPath}", f"{MN5_transfer_host}:{destAbsPath}"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    else :
        subprocess.run(["scp", f"{inpAbsPath}", f"{MN5_transfer_host}:{destAbsPath}"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    
# On local machine, checks if directory exists. If not, creates it
def isdir_create(dirname :str) -> bool :
    if not(os.path.isdir(dirname)):
        os.mkdir(dirname)
        return False
    return True

# On remote machine, checks if directory exists. If not, creates it
def isdir_create_remote(dirname :str) -> None :
    cmd = [f"""if [ ! -d "{dirname}" ]; then mkdir {dirname}; fi"""]
    exec_remote_cmds(cmd)