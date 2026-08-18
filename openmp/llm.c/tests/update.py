from tqdm import tqdm
from os.path import join
from os import listdir
from config import remoteMN5_scriptsFolder, remoteFolder, localFolder, utilsFolder, \
    MN5_scriptsFolder, devDataShakespeare, devDataStories, remoteDevDataShakespeare, remoteDevDataStories
from data.utils.utils import transfer_to_remote

def update():
    OBJS=[
    (join(localFolder, "Makefile"), join(remoteFolder, "Makefile")),
    (join(localFolder, "train_gpt2.c"), join(remoteFolder, "train_gpt2.c")),
    (join(utilsFolder, "user-functions.dat"), join(remoteFolder, "user-functions.dat")),
    (join(utilsFolder, "extrae-testingPipeline.xml"), join(remoteFolder, "extrae-testingPipeline.xml")),
    (join(MN5_scriptsFolder, "trace.sh"), join(remoteMN5_scriptsFolder, "trace.sh")),
    (join(MN5_scriptsFolder, "ovniemu.sh"), join(remoteMN5_scriptsFolder, "ovniemu.sh"))
    ]
    
    for obj in tqdm(OBJS):
        transfer_to_remote(obj[1], obj[0])
        
    
def initialize():
    OBJS=[
    (join(localFolder, "gpt2_124M_bf16.bin"), join(remoteFolder, "gpt2_124M_bf16.bin")),
    (join(localFolder, "gpt2_124M.bin"), join(remoteFolder, "gpt2_124M.bin")),
    (join(localFolder, "gpt2_124M_debug_state.bin"), join(remoteFolder, "gpt2_124M_debug_state.bin")),
    (join(localFolder, "gpt2_tokenizer.bin"), join(remoteFolder, "gpt2_tokenizer.bin")),
    ]
    
    for obj in tqdm(OBJS):
        transfer_to_remote(obj[1], obj[0])
    
    tinyShakespeareData = ["tiny_shakespeare.txt", "tiny_shakespeare_train.bin", "tiny_shakespeare_val.bin"]
    tinyStoriesData = []
    
    tinyShakespeareAvailableData = listdir(devDataShakespeare)
    tinyStoriesAvailableData = listdir(devDataStories)
    
    toTransferTo = []
    toTransferFrom = []
    for fileName in tinyShakespeareData:
        if fileName in tinyShakespeareAvailableData:
            toTransferTo.append(join(remoteDevDataShakespeare, fileName))
            toTransferFrom.append(join(devDataShakespeare, fileName))
    for fileName in tinyStoriesData:
        if fileName in tinyStoriesAvailableData:
            toTransferTo.append(join(remoteDevDataStories, fileName))
            toTransferFrom.append(join(devDataStories, fileName))
            
    for i in tqdm(range(len(toTransferTo))):
        transfer_to_remote(toTransferTo[i], toTransferFrom[i])
        
    transfer_to_remote(remoteFolder, join(localFolder, "llmc"), True)
    update()


if __name__ == "__main__":
    update()