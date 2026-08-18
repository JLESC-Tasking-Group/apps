import os
from sys import argv
import numpy as np
from typing import List

eps=2.1e-6
expectedLoss = [5.356192,
4.301084,
4.623322,
4.600488,
4.61679,
4.231492,
3.754268,
3.652406,
4.18363,
4.199315,
4.288379,
3.558853,
3.730747,
4.159209,
3.886536,
3.764807,
4.142973,
3.962871,
3.796139,
3.371689,
3.880784,
4.198482,
4.425916,
3.685767,
3.642242,
3.729666,
3.549569,
3.339429,
4.338737,
3.812686,
4.027639,
4.114108,
3.574935,
4.365806,
4.515868,
4.433772,
4.097108,
3.739621,
4.611547,
3.970719,
4.37773,
4.382721,
3.619619,
3.513283,
3.715509,
3.641472,
3.980987,
3.729475,
3.797705,
4.098539,
3.718207,
3.979465,
3.660579,
3.713463,
3.548756,
3.653562,
4.633749,
4.778025,
4.139521,
4.197925,
4.194864,
3.442871,
3.547137,
3.921847,
4.413553,
4.173967,
3.725983,
4.204272,
3.55733,
3.255034,
2.718048,
3.339985,
2.281523,
4.003077,
3.99,
2.42976,
3.575534,
3.632228,
4.499602,
3.627292
]



def are_equal(arr1 : List[float], arr2 :List[float]) -> bool:
    N = min(len(arr1), len(arr2))
    isEq = True
    for i in range(N):
        if abs(arr1[i] - arr2[i]) > eps:
            print(f"\033[93mUnequal loss for step {i},\tExpected {arr1[i]} but got {arr2[i]}\033[0m")
            isEq = False
    return isEq
    
if len(argv) < 2:
    print("Missing arguments")
    exit(1)

# Check for inputs existence 
filePath = argv[1]
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
    
    
if "train loss" in identifiers:
    maxValuesLen = 0
    isModelValid = True
    for fileValues in data["train loss"]:
        maxValuesLen = max(maxValuesLen, len(fileValues))
        isModelValid = are_equal(expectedLoss, fileValues) if isModelValid else False
    
    if isModelValid:
        if maxValuesLen < 4:
            print("\033[93mWARN : Too few loss values to ensure model validity\033[0m")
        print("\033[92m============== MODEL VALID ==============\033[0m")
    else:
        print("\033[91m============== MODEL NOT VALID ==============\033[0m")
        
        
else:
    print("No loss values in output")
    print("\033[93mWARN : Unable to check model validity\033[0m")