import matplotlib.pyplot as plt
from sys import argv
import numpy as np
import os
from textwrap import wrap
from typing import List

forgetFirstIteration = True

if len(argv) < 3:
    print("Missing arguments")
    exit(1)
    


def time_per_iteration(data :List[List[int]], filePath :str, nbThread :int, label :str) -> plt.Figure :
    if forgetFirstIteration:
        data = data[:,1:]
    fig = plt.figure()
    fig.set_figheight(5.5)
    X = [[i for i in range(len(data[j]))] for j in range(nbFiles)]
    Y = [data[i,:] for i in range(nbFiles)]
    for x,y in zip(X,Y):
        plt.plot(x,y)
    plt.grid(fig)
    plt.xlabel("Iteration id")
    plt.ylabel("Time taken per iteration (ms)")
    plt.title("\n".join(wrap(f"{label} runtime per iteration for {os.path.basename(filePath)[:-4]} implementation on {nbThread} threads/cpus (all {nbFiles} executions)", 60)))
    return fig

def mean_time_per_iteration(data :List[List[int]], filePath :str, nbThread :int, label :str) -> plt.Figure :
    if forgetFirstIteration:
        data = data[:,1:]
    fig = plt.figure()
    fig.set_figheight(5.5)
    X = [i for i in range(len(data[0]))]
    Y = [np.mean(data[:,i]) for i in range(len(data[0]))]
    meanRuntime = np.mean(Y)
    plt.plot(X,Y, label="Mean runtime per executions")
    plt.plot(X, [meanRuntime] * len(X), label=f"Mean runtime per {label} ({int(meanRuntime)} ms)")
    plt.grid()
    plt.xlabel("Iteration id")
    plt.ylabel(f"Time taken per {label} (ms)")
    plt.title("\n".join(wrap(f"Mean runtime per {label} for {os.path.basename(filePath)[:-4]}\n implementation on {nbThread} threads/cpus (on {nbFiles} executions)", 60)))
    plt.legend()
    print()
    print(f"Mean runtime per {label} :", np.round(meanRuntime, 0), "ms")
    print(f"{label} runtime standard deviation : {np.round(np.mean(np.std(data, 0)),1)} ms\n")
    return fig

def time_in_matmul_per_iteration(data :List[List[int]], nbIterations :int, filePath :str, nbThread :int, label :str) -> plt.Figure :
    if forgetFirstIteration:
        data = data[:,nbIterations:]
    fig = plt.figure()
    fig.set_figheight(5.5)
    X = [i for i in range(len(data[0]))]
    Y = [np.mean(data[:,i]) for i in range(len(data[0]))]
    Xsub = [i for i in range(int(len(X)//nbIterations))]
    Ysub = [[y for y in Y[i:i+nbIterations]] for i in range(0, len(Y), nbIterations)]
    # mean1 = np.mean(Y)
    maxTime = [np.max(arr) for arr in Ysub] # max time per iteration passed in a single matmul
    totTime = [np.sum(arr) for arr in Ysub] # sum time per iteration passed in matmul
    plt.plot(Xsub, maxTime, label=f"Max time taken by a single {label}")
    plt.plot(Xsub, totTime, label=f"Time passed in {label} per iteration")
    # plt.plot(X, [mean1] * len(X), label=f"Mean runtime per {label} ({int(mean1)} ms)")
    plt.grid()
    plt.xlabel("Iteration id")
    plt.ylabel(f"Time (ms)")
    plt.title("\n".join(wrap(f"Runtime in {label} for {os.path.basename(filePath)[:-4]}\n implementation on {nbThread} threads/cpus (on {nbFiles} executions)", 60)))
    plt.legend()
    print()
    print(f"Time in matmul {label} : {np.round(np.mean(totTime), 0)} ms")
    print(f"{label} total runtime standard deviation : {np.round(np.mean(np.std(data, 0)), 1)} ms")
    return fig


# Check for inputs existence 
filePath = argv[1]
nbThread = argv[2]
testsFolder = os.path.dirname(os.path.dirname(__file__))

version = os.path.basename(filePath).split("-")[2]
versionFolder = os.path.join(testsFolder, version)
if not(os.path.isdir(versionFolder)):
    os.mkdir(versionFolder)
graphVersionFolder = os.path.join(versionFolder, "graphs")
if not(os.path.isdir(graphVersionFolder)):
    os.mkdir(graphVersionFolder)
ouputGraphsBaseName = os.path.join(graphVersionFolder, os.path.basename(filePath)[:-4])
    

# Retrieve data from csv file
with open(filePath, mode="r") as f:
    txt = f.readlines()

header = txt[0]
    
identifiers = header.split(" | ")[:-1]
nbFiles = len(txt[2].split(" "))-1
# identifier -> file -> it
data ={identifier : [[] for i in range(nbFiles)] for identifier in identifiers}
currentIdentifierId = -1
for i, line in enumerate(txt[1:]):
    if line =="\n":
        continue
    if currentIdentifierId+1 != len(identifiers) and identifiers[currentIdentifierId+1] in line:
        currentIdentifierId+=1
        continue
    

    splittedLine = map(float, line[:-2].split(" "))
       
        
    for j, elmt in enumerate(splittedLine):
        data[identifiers[currentIdentifierId]][j].append(elmt)

for identifier in identifiers:
    data[identifier] = np.array(data[identifier])



# Draw runtime per iteration for all iteration
if "iteration runtime" in data.keys():
    outputFullPath = ouputGraphsBaseName + "-all.png"
    fig = time_per_iteration(data["iteration runtime"], filePath, nbThread, "Iteration")
    fig.savefig(outputFullPath)
   
# Draw average runtime of all execution, per iteration
if "iteration runtime" in data.keys():
    outputFullPath = ouputGraphsBaseName + "-mean.png"
    fig = mean_time_per_iteration(data["iteration runtime"], filePath, nbThread, "iteration")
    fig.savefig(outputFullPath)
    
# Print average runtime for model forward pass
if "forward runtime" in data.keys():
    if forgetFirstIteration:
        forwardData = data["forward runtime"][:,1:]
    else:
        forwardData = data["forward runtime"]
    print("Mean runtime per forward pass :", np.round(np.mean(forwardData), 1), "ms")
    print(f"Forward pass runtime standard deviation : {np.round(np.mean(np.std(forwardData, 0)),1)} ms\n")

# Print average runtime for model backward pass
if "backward runtime" in data.keys():
    if forgetFirstIteration:
        backwardData = data["backward runtime"][:,1:]
    else:
        backwardData = data["backward runtime"]
    print("Mean runtime per backward pass :", np.round(np.mean(backwardData), 1), "ms")
    print(f"Backward pass runtime standard deviation : {np.round(np.mean(np.std(backwardData, 0)),1)} ms\n")

# Draw runtime in matmul forward
if "matmul forward runtime" in data.keys() and "iteration runtime" in data.keys():
    outputFullPath = ouputGraphsBaseName + "-matmul-forward.png"
    nbIterations = int(len(data["matmul forward runtime"][0]) / len(data["iteration runtime"][0]))
    fig = time_in_matmul_per_iteration(data["matmul forward runtime"], nbIterations, filePath, nbThread, "matmul forward")
    fig.savefig(outputFullPath)

# Draw runtime in matmul backward
if "matmul backward runtime" in data.keys() and "iteration runtime" in data.keys():
    outputFullPath = ouputGraphsBaseName + "-matmul-backward.png"
    nbIterations = int(len(data["matmul backward runtime"][0]) / len(data["iteration runtime"][0]))
    fig = time_in_matmul_per_iteration(data["matmul backward runtime"], nbIterations, filePath, nbThread, "matmul backward")
    fig.savefig(outputFullPath)
    
# Ompss kernels in forward pass
labels = ["ompss Encoder",
"ompss Layernorm",
"ompss Matmul",
"ompss Attention",
"ompss Gelu",
"ompss Residual",
"ompss Softmax",
"ompss Crossentropy"]
L=12
itInFirstRun = {
    "ompss Encoder" : 1, 
    "ompss Layernorm" : 2*L+1, 
    "ompss Matmul": 4*L,
    "ompss Attention": L,
    "ompss Gelu": L,
    "ompss Residual": 2*L,
    "ompss Softmax": 1,
    "ompss Crossentropy": 1
}
meanOut = []
totOut = []
for label in labels:
    if label in data.keys():
        labelData = data[label]
        nbIterations = int(len(labelData[0]) / itInFirstRun[label])
        print(nbIterations)
        #print(labelData[0])
        if forgetFirstIteration:
            labelData = labelData[:,itInFirstRun[label]:]
        
        print("\nAverage time in", label, np.round(np.mean(labelData), 6), "ms")
        meanOut.append(np.round(np.mean(labelData), 6))
        print("Total time in", label, np.round(np.sum(labelData)/(nbIterations-(1 if forgetFirstIteration else 0)), 6), "ms")
        totOut.append(np.round(np.sum(labelData)/(nbIterations-(1 if forgetFirstIteration else 0)), 6))
        
print("Mean :", "\t".join(map(str,meanOut)))
print("Total :", "\t".join(map(str,totOut)))

print()
