from tqdm import tqdm
import os
from sys import argv



if len(argv) < 3:
    print("No name given for output file or No files to convert have been given in argument")
    exit(1)

#args = argv[1].split(" ")
# Times to retrieve needs to be between the 2 tokens of an identifier
identifiers = {
    "iteration runtime" : ("@iteration runtime : ", " ms@"),
    "matmul forward runtime" : ("@matmul forward runtime : ", " ms@"),
    "matmul backward runtime" : ("@matmul backward runtime : ", " ms@"),
    "forward runtime" : ("@forward runtime : ", " ms@"),
    "backward runtime" : ("@backward runtime : ", " ms@"),
    "total runtime" : ("@total runtime : ", " ms@"),
    "initialization runtime" : ("@initialization runtime : ", " ms@"),
    "train loss" : ("@train loss : ", " ms@"),
    "ompss Encoder" : ("@Encoder : ", " ms@"),
    "ompss Layernorm" : ("@Layernorm : ", " ms@"),
    "ompss Matmul" : ("@Matmul : ", " ms@"),
    "ompss Attention" :("@Attention : ", " ms@"),
    "ompss Gelu" : ("@Gelu : ", " ms@"),
    "ompss Residual" : ("@Residual : ", " ms@"),
    "ompss Softmax" : ("@Softmax : ", " ms@"),
    "ompss Crossentropy" : ("@Crossentropy : ", " ms@")
}
testsDir :str = os.path.dirname(os.path.dirname(__file__))
outFolder :str = os.path.join(testsDir, "data/out")
csvFolder :str = os.path.join(testsDir, "data/csv")
outputFile = os.path.join(csvFolder, argv[1])

if not(os.path.isdir(csvFolder)):
    os.mkdir(csvFolder)


# Check and get all the absolute path for the given files
files = []
for file in argv[2:]:
    if os.path.isfile(file): # If given path is absolute
        files.append(file)
        
    elif os.path.isfile(os.path.join(outFolder, file)):
        files.append(os.path.join(outFolder, file))
       
    else:
        print("File", file, "not found") 

# fileId -> identifierDict -> times
data = [{identifier : [] for identifier in identifiers} for i in range(len(files))]
usefulIdentifiers = set()
print("Reading...")
for i in tqdm(range(len(files))):
    with open(files[i], mode="r") as f:
        lines = f.readlines()
    if i == 0:
        for line in lines:
            for identifier in identifiers:
                m1 = line.find(identifiers[identifier][0])
                m2 = line.find(identifiers[identifier][1])
                if m1 == -1 or m2 == -1 or m1 >= m2:
                    continue
                timeIdx1 = m1 + len(identifiers[identifier][0])
                timeIdx2 = m2
                print("--------------")
                print(timeIdx1, timeIdx2, m1, m2)
                print("\n\n ->",i, line, "<- \n\n")
                print(timeIdx1,timeIdx2)

                time = float(line[timeIdx1:timeIdx2])
                data[i][identifier].append(time)
                usefulIdentifiers.add(identifier)
    else :
        for line in lines:
            for identifier in usefulIdentifiers:
                m1 = line.find(identifiers[identifier][0])
                m2 = line.find(identifiers[identifier][1])
                if m1 == -1 or m2 == -1 or m1 >= m2:
                    continue
                timeIdx1 = m1 + len(identifiers[identifier][0])
                timeIdx2 = m2

                time = float(line[timeIdx1:timeIdx2])
                data[i][identifier].append(time)

FlippedData ={identifier :[[0 for j in range(len(files))] for i in range(len(data[0][identifier]))] for identifier in identifiers}
print("Writting...")
with open(outputFile, mode="w") as f:
    for identifier in usefulIdentifiers:
        f.write(identifier + " | ")

    for identifier in tqdm(usefulIdentifiers):
        f.write("\n" + identifier + "\n")
        for i in range(len(data[0][identifier])):
            for j in range(len(files)):
                f.write(str(data[j][identifier][i]) + " ")
            f.write("\n")



# identifier1 | identifier2 | identifier3 ....
# identifier1
# 213, 213, 432, 232 <- iteration 1, for each files
# 432, 4324, 324, 12 <- it 2 for each file
# identifier2
# etc...
            
            
        
        

# def extract_time(line:str)->str|None:
#     try:
#         startIdx = line.index("<") + 1
#         endIdx = line.index(">", startIdx)
#     except ValueError:
#         return None
#     if line[startIdx:endIdx] == "|endoftext|" :
#         return None
#     return line[startIdx:endIdx]


# # Path to files is /dev/data/outputs/computationType/csv-or-raw/
# computationTypes = ["sequential", "openmp", "openmpv"]
# datasets = ["tinystories", "tinyshakespeare"]

# unformattedFolderPath = dirname(dirname(__file__)) + "/dev/data/outputs/{}/{}/raw"
# allRawFolders = []
# for dataset in datasets:
#     for computationType in computationTypes:
#         baseFolderPath = unformattedFolderPath.format(dataset, computationType)
#         allRawFolders.extend([f"{baseFolderPath}/{subFolder}/" for subFolder in listdir(baseFolderPath)])



# for rawFolderPath in allRawFolders:
#     csvEqFolder = rawFolderPath.replace("raw", "csv")
#     if not(exists(csvEqFolder)):
#         mkdir(csvEqFolder)

#     files = [file for file in listdir(rawFolderPath) if (".out" == file[-4:])]
#     for file in files:
#         csvFileName = csvEqFolder + file[:-4] + ".csv"
#         if isfile(csvFileName): # CSV already computed
#             continue
#         with open(rawFolderPath + "/" + file, "r") as f:
#             data = f.readlines()
        
#         times = [extract_time(line) for line in data]
#         times = [elmt for elmt in times if elmt is not None]
#         print(file)
#         print(times)
#         print(not(isfile(f"{file[:-4]}.csv")))

#         with open(csvFileName, "w") as f:
#             csvWritter = csv.writer(f, delimiter='\n', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#             csvWritter.writerow(times)
