# You have to setup your own configuration, and to rename this file as 'config.py' instead of 'config_example.py'
import os
from typing import List

##########################################################################

MN5_exec_host = "username@glogin1.bsc.es" # Host for execution purposes
# If you have configured a hostname in your ~/.ssh/config as follow :
"""
host glogin1
    hostname glogin1.bsc.es
    user username
"""
# You can directly set MN5_remote_machine to glogin1

##########################################################################

MN5_transfer_host = "username@transfer1.bsc.es" # Host for data transfer
# You can configure it in the same way than the execution host above

##########################################################################

# Path to the MN5 directory in which the code will be installed
# Please use a different directory name than gpt2 so my tests do not clash with yours
# It is mandatory to stay in the bsc15 directory
MN5_remote_folder = "/gpfs/projects/bsc15/myGPT_directory"

##########################################################################

sbatch_queue = "gp_bsccs" # Queue to use for sbacth

##########################################################################

# For every instrumentation, set tot True if you want the paraver traces to be downloaded onto your local machine
downloadTrace = True 

##########################################################################

# Configure local and remote directories
# I recommand to let it untouched 
testsFolder :str                = os.path.dirname(__file__)
localFolder :str                = os.path.dirname(testsFolder)
dataFolder :str                 = os.path.join(testsFolder, "data") 
csvFolder :str                  = os.path.join(dataFolder, "csv") 
analysisFolder :str             = os.path.join(testsFolder, "analysis")
outFolder :str                  = os.path.join(dataFolder, "out")
tmpFolder :str                  = os.path.join(dataFolder, "tmp")
utilsFolder :str                = os.path.join(dataFolder, "utils")
MN5_scriptsFolder :str          = os.path.join(localFolder, "MN5_scripts")

sequentialFolder :str           = os.path.join(testsFolder, "sequential")
openmpFolder :str               = os.path.join(testsFolder, "openmp")
openmpvFolder :str              = os.path.join(testsFolder, "openmpv")

sequentialTraceFolder :str      = os.path.join(sequentialFolder, "traces")
openmpTraceFolder :str          = os.path.join(openmpFolder, "traces")
openmpvTraceFolder :str         = os.path.join(openmpvFolder, "traces")
sequentialgraphsFolder :str     = os.path.join(sequentialFolder, "graphs")
openmpgraphsFolder :str         = os.path.join(openmpFolder, "graphs")
openmpvgraphsFolder :str        = os.path.join(openmpvFolder, "graphs")

dev :str                        = os.path.join(localFolder, "dev")
devData :str                    = os.path.join(dev, "data")
devDataShakespeare :str         = os.path.join(devData, "tinyshakespeare")
devDataStories :str             = os.path.join(devData, "tinystories")


remoteFolder :str               = MN5_remote_folder
remoteOutputsFolder :str        = os.path.join(remoteFolder, "outputs")
remoteTmpFolder :str            = os.path.join(remoteOutputsFolder, "tmp")
remoteTracesFolder :str         = os.path.join(remoteOutputsFolder, "traces")
remoteMN5_scriptsFolder :str    = os.path.join(remoteFolder, "MN5_scripts")

remoteDev :str                  = os.path.join(remoteFolder, "dev")
remoteDevData :str              = os.path.join(remoteDev, "data")
remoteDevDataShakespeare :str   = os.path.join(remoteDevData, "tinyshakespeare")
remoteDevDataStories :str       = os.path.join(remoteDevData, "tinystories")

allRemotesPaths :List[str] = [
    remoteFolder,
    remoteOutputsFolder,
    remoteTmpFolder,
    remoteTracesFolder,
    remoteMN5_scriptsFolder,
    remoteDev,
    remoteDevData,
    remoteDevDataShakespeare,
    remoteDevDataStories
]

allLocalPaths :List[str] = [
    testsFolder,
    dataFolder,
    csvFolder,
    analysisFolder,
    outFolder,
    tmpFolder,
    sequentialFolder,
    openmpFolder,
    openmpvFolder,
    sequentialTraceFolder,
    openmpTraceFolder,
    openmpvTraceFolder,
    sequentialgraphsFolder,
    openmpgraphsFolder,
    openmpvgraphsFolder,
    dev,
    devData,
    devDataShakespeare,
    devDataStories
]


versions :List[str] = ["sequential", "openmp", "openmpv"]
datasets :List[str] = ["tinyshakespeare"]
binaries :dict[str, str] = {
        "sequential": "train_gpt2_seq",
        "openmp"    : "train_gpt2_openmp",
        "openmpv"   : "train_gpt2_openmpv" 
    }
instrumentationTypes :dict[str, str] = {
    "sequential": "None",
    "openmp"    : "extrae",
    "openmpv"   : "ovni" 
}

traceFolders :dict[str, str] = {
    "sequential": sequentialTraceFolder,
    "openmp"    : openmpTraceFolder,
    "openmpv"   : openmpvTraceFolder 
}