import os
import argparse
from typing import List
import subprocess
import time
from data.utils.utils import exec_local_cmds, exec_remote_cmds, isdir_create,transfer_from_remote, transfer_to_remote
try :
    from config import *
except:
    print("Config file not found or wrongly written.\nSee tests/config_example.py for futher informations.")

remoteExecutionMachine :str = MN5_exec_host
remoteTransferMachine :str = MN5_transfer_host
queueName :str = sbatch_queue

##############################################################
#########################   UTILS   ##########################
##############################################################


# Clean up on error
def clean_up(testId :int, jobIds :List[int] | None, version :str) -> None: 
    # Clean remote
    cleanCmd = " && ".join(
        [f"cd {remoteFolder}",
        f"rm -rf MN5_scripts/test-{testId}-*",
        f"rm -f outputs/test-{testId}-"])
    
    if jobIds is not None:
        cancelCmd = " && ".join([f"scancel {jobId}" for jobId in jobIds])
        rmExtra = " && ".join([f"rm -f outputs/tmp/{jobId} && rm -rf outputs/tmp/node-{jobId}" for jobId in jobIds])
    else:
        cancelCmd = "echo 'A'"
        rmExtra = "echo 'B'"
    subprocess.run(["ssh", "glogin1", f"{cancelCmd} && {cleanCmd} && {rmExtra}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, text=True)
    
    # Clean local
    rmCmds = ["rm",
              "-rf",
             f"{outFolder}/test-{testId}-*",
             f"{csvFolder}/test-{testId}-*",
             f"{tmpFolder}/test-{testId}-*",
             f"{traceFolders[version]}/test-{testId}-*",
             f"{testsFolder}/{version}/graphs/test-{testId}-*"
    ]
    subprocess.run(rmCmds, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, text=True)
    

    
##############################################################
########################   CHECK UP   ########################
##############################################################


# Check for the necessary folders

isdir_create(dataFolder)
isdir_create(analysisFolder)
isdir_create(tmpFolder)
isdir_create(outFolder)

isdir_create(sequentialFolder)
isdir_create(openmpFolder)
isdir_create(openmpvFolder)

isdir_create(sequentialTraceFolder)
isdir_create(openmpTraceFolder)
isdir_create(openmpvTraceFolder)
isdir_create(ompssTraceFolder)
isdir_create(sequentialgraphsFolder)
isdir_create(openmpgraphsFolder)
isdir_create(openmpvgraphsFolder)
isdir_create(ompssgraphsFolder)


assert(len(traceFolders) == len(versions))
assert(len(versions) == len(binaries))
assert(len(versions) == len(instrumentationTypes))

##############################################################
######################   ARG PARSING   #######################
##############################################################


def check_args(args:argparse.Namespace) -> None:
    jobs:int = args.jobs
    threads:int = args.threads
    instrument :bool = args.instrument
    version :str = args.version
    isCorrect = True

    if jobs < 0 and instrument:
        print("ERROR : Expecting number of jobs to be >= 0 when instrument=True")
        isCorrect = False
    
    if jobs <= 0 and not(instrument):
        print("ERROR : Expecting number of jobs to be > 0 when instrument=False")
        isCorrect = False

    if threads < 1 or threads > 112:
        print("ERROR : Expecting threads to be > 0")
        isCorrect = False

    if version == "sequential" and instrument:
        print("ERROR : No instrumentation available for sequential version")
        isCorrect = False
        
    if jobs > 15:
        ans = input(f"You are about to launch many jobs ({jobs}), each one on a different node. Launch anyway ? (Y/N) ")
        if not(ans.upper()) in ["YES", "YE", "Y"]:
            print("Exiting")
            isCorrect = False
            

    if not(isCorrect):
        exit(1)
   


parser = argparse.ArgumentParser(
                    prog='launcher.py',
                    description='Test launcher',
                    epilog='')


parser.add_argument('-d', '--dataset', type=str, default="tinyshakespeare", choices=datasets, help="Dataset to use (only tinyshakespeare is available right now)")
parser.add_argument('-i', '--instrument', action='store_true', default=False, help="Enable instrumentation on a new job with ovni or extrae. This job is the only one to be launched with instrumentation, contrary to the N other jobs")
parser.add_argument('-N', '--jobs', type=int, default=1, help="Number of executions to run. If instrumentation is enabled, can be set to 0, otherwise, should be >=1")
parser.add_argument('-t', '--threads', type=int, required=True, help="Number of threads or cpus to use. We use 1 thread per CPU")   
parser.add_argument('-v', '--version', type=str, choices=versions, required=True, help="Implementation version to use")  

args:argparse.Namespace = parser.parse_args()

check_args(args)

dataset :str = args.dataset
instrument :bool = args.instrument
jobs :int = args.jobs
threads :int = args.threads
version :str = args.version



##############################################################
####################   GENERATE SCRIPTS   ####################
##############################################################   
        

def get_test_id(outFolder :str) -> int:
        
    files :List[List[str]] = [fileName.split("-") for fileName in os.listdir(tmpFolder)] 
    testIds : List[int] = [int(fileName[1]) for fileName in files if len(fileName) > 1 and fileName[1].isdigit()]
    
    return 1 if len(testIds) == 0 else (max(testIds)+1)
    
def get_script_name(testId :int, dataset: str, instrument :bool, version :str) -> str :
    return f"test-{testId}-{version}-{dataset}{'-i' if instrument else ''}.sh"

def get_compilation_cmd(instrument :bool, version :str) -> str :
    if instrument:
        return f"make {binaries[version]}-i"
    return f"make {binaries[version]}"
        

def get_launch_script(testId :int, dataset :str, instrument :bool, threads :int, version :str) -> str:
    common:str = """\
#!/bin/bash
#SBATCH --job-name={jobName}
#SBATCH --chdir=.
#SBATCH --output=outputs/{stdOutputName}%j.out
#SBATCH --error=outputs/{errOutputName}%j.err
#SBATCH --cpus-per-task={nbCpu}
#SBATCH --ntasks=1
#SBATCH --time=00:02:00
#SBATCH --qos={queueName}
#SBATCH --exclusive
"""

    previous :str = "\n\n"

    launch :str = "\ntime "
    post :str = "\n"

    if instrumentationTypes[version] == "ovni" and instrument:
        previous+="export OVNI_TMPDIR=${TMPDIR}\n"
        previous += """if [ ! -d "${TMPDIR}" ]; then mkdir ${TMPDIR}; fi\n""".format(TMPDIR="{TMPDIR}")
        if version == "ompss":
            # launch += "OMP_OVNI=2 NANOS6_CONFIG_OVERRIDE=\"version.instrument=ovni,instrument.ovni.level=1,monitoring.enabled=true\" "
            launch += "NODES_ITER_PRINT=1 NODES_OVNI=2 NOSV_CONFIG_OVERRIDE=\"instrumentation.version=ovni\" "
        elif version == "openmpv":
            launch += "OMP_OVNI=2 NOSV_CONFIG_OVERRIDE=\"instrumentation.version=ovni\" "

    if version == "ompss":
        previous += "export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}\n"
        launch += f"taskset -c 0-{threads-1} " 
    
    if version in ["openmp", "openmpv"]:
        previous += "export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}\n"
        launch += f"OMP_NUM_THREADS={threads} taskset -c 0-{threads-1} "
    
    if version == "openmp" and instrument:
        launch += "./MN5_scripts/trace.sh "
        previous += """if [ ! -d "${TMPDIR}" ]; then mkdir ${TMPDIR}; fi\n""".format(TMPDIR="{TMPDIR}")
        previous += """if [ ! -d "{remoteTmpFolder}/${SLURM_JOBID}" ]; then mkdir {remoteTmpFolder}/${SLURM_JOBID}; fi\n""".format(remoteTmpFolder=remoteTmpFolder, SLURM_JOBID="{SLURM_JOBID}")
        previous += "cp {remoteFolder}/extrae-testingPipeline.xml {remoteTmpFolder}/${SLURM_JOBID}/extrae.xml\n".format(remoteFolder=remoteFolder, remoteTmpFolder=remoteTmpFolder, SLURM_JOBID="{SLURM_JOBID}")
        previous += """sed -i "s+TEST-ID-DIRECTORY+{remoteTmpFolder}/${SLURM_JOBID}+g" {remoteTmpFolder}/${SLURM_JOBID}/extrae.xml\n""".format(remoteTmpFolder=remoteTmpFolder, SLURM_JOBID="{SLURM_JOBID}")
        previous += """sed -i "s+TMPDIR+${TMPDIR}+g" {remoteTmpFolder}/${SLURM_JOBID}/extrae.xml\n""".format(TMPDIR="{TMPDIR}", remoteTmpFolder=remoteTmpFolder, SLURM_JOBID="{SLURM_JOBID}")
        post += "mv ${TMPDIR}/TRACE.mpits {remoteTmpFolder}/${SLURM_JOBID}/TRACE.mpits\n".format(TMPDIR="{TMPDIR}", remoteTmpFolder=remoteTmpFolder, SLURM_JOBID="{SLURM_JOBID}")
        post += "mv ${TMPDIR}/set-0 {remoteTmpFolder}/${SLURM_JOBID}/".format(TMPDIR="{TMPDIR}", remoteTmpFolder=remoteTmpFolder, SLURM_JOBID="{SLURM_JOBID}")
    
    
    launch += f"./{binaries[version]}"
    if instrument:
        launch += "-i"
        previous += f"echo $SLURMD_NODENAME > {remoteTmpFolder}/node-$SLURM_JOBID\n"
    launch += " && >&2 echo 'train_gpt2 : EXIT_SUCCESS' || >&2 echo 'train_gpt2 : EXIT_FAILURE'"



    common = common.format(jobName=version+"_job_"+str(testId),
                  stdOutputName=f"{get_script_name(testId, dataset, instrument, version)[:-3]}-",
                  errOutputName=f"{get_script_name(testId, dataset, instrument, version)[:-3]}-",
                  nbCpu=threads,
                  queueName=queueName)
    

    script :str = common + previous + launch + post
    return script

def get_sbatch_cmd(testId :int, dataset :str, instrument :bool, jobs :int, version :str) -> str :
    cmd :str = "sbatch "

    if jobs > 1:
        cmd += f"--array=1-{jobs} "
    elif jobs == 0:
        return ""
    
    cmd += f"-A bsc15 -q {queueName} --exclusive {remoteMN5_scriptsFolder}/{get_script_name(testId, dataset, instrument, version)}"
    return cmd


# Create sbatch target script and inline sbatch command
testId :int = get_test_id(outFolder)
print(f"Running test nº{testId}")
jobIds = None

try :
    scripts = []
    sbatchs = []
    scriptPaths = []
    compilations = []
    scriptsRawNames = []
    scriptRemotePaths = []
    if jobs > 0:
        scripts.append(get_launch_script(testId, dataset, False, threads, version))
        sbatchs.append(get_sbatch_cmd(testId, dataset, False, jobs, version))
        scriptPaths.append(f"{tmpFolder}/{get_script_name(testId, dataset, False, version)}")
        compilations.append(get_compilation_cmd(False, version))
        scriptsRawNames.append(get_script_name(testId, dataset, False, version))
        scriptRemotePaths.append(f"{remoteMN5_scriptsFolder}/{scriptsRawNames[-1]}")



    if instrument:
        instrumentedTestInternalId = len(scripts)
        scripts.append(get_launch_script(testId, dataset, True, threads, version))
        sbatchs.append(get_sbatch_cmd(testId, dataset, True, 1, version))
        scriptPaths.append(f"{tmpFolder}/{get_script_name(testId, dataset, True, version)}")
        compilations.append(get_compilation_cmd(True, version))
        scriptsRawNames.append(get_script_name(testId, dataset, True, version))
        scriptRemotePaths.append(f"{remoteMN5_scriptsFolder}/{scriptsRawNames[-1]}")



    # Write .sh script for sbatch
    for scriptPath, script in zip(scriptPaths, scripts):
        with open(scriptPath, mode="w") as f:
            f.write(script)

    ##############################################################
    ######################   SCRIPTS EXEC   ######################
    ##############################################################

    # Get the node on which an instrumented test is running
    def get_exec_node(jobId :int, squeueOutput :List[str]) -> str :
        for process in squeueOutput[:-1]:
            splittedLine = process.split(" ")
            if str(jobId) in splittedLine[1] and splittedLine[3] != "0:00":
                return splittedLine[2] 
        return None
        

    marker :str = f"--job-from-test-{testId}"
    commands = [
        f"cd {remoteFolder}",
        "module load ompss-2/git",
        "module load extrae",
        #"export OVNI_TMPDIR=/home/bsc/bsc737963",
        #"module load papi"
        #"module load ompss-2/git",
        f"export GPT_HOME={remoteFolder}"         
        ]
    # module load ompss-2/git
    # export NOSV_HOME=/home/bsc/bsc015728/instr/nos-v/install
    # clang -fopenmp=libompv t1.c -Wl,-rpath,/home/bsc/bsc015728/instr/install/lib
    # Expect :
    # linux-vdso.so.1 (0x00007ffc1795c000)
	# /opt/.snoopy/lib/libsnoopy.so (0x00007f6512f90000)
	# libnosv.so.0 => /home/bsc/bsc015728/instr/nos-v/install//lib/libnosv.so.0 (0x00007f6512f5e000)
	# libompv.so => /home/bsc/bsc015728/instr/install/lib/libompv.so (0x00007f6512e1a000)
	# libc.so.6 => /lib64/libc.so.6 (0x00007f6512c03000)
	# libnuma.so.1 => /lib64/libnuma.so.1 (0x00007f6512bf5000)
	# libovni.so.1 => /apps/GPP/PM/ovni/git/lib/libovni.so.1 (0x00007f6512be4000)
	# libm.so.6 => /lib64/libm.so.6 (0x00007f6512b07000)
	# /lib64/ld-linux-x86-64.so.2 (0x00007f6512fa1000)

    if jobs > 0:
        commands.append(f"rm -f {binaries[version]}")
    if instrument:
        commands.append(f"rm -f {binaries[version]}-i")



    for i in range(len(scripts)):
        commands.append(f"chmod 744 {scriptRemotePaths[i]}")
        commands.append(compilations[i])
        commands.append(sbatchs[i])


    # Copy sbatch target script on MN5
    for scriptPath, scriptRemotePath in zip(scriptPaths, scriptRemotePaths):
        transfer_to_remote(f"{scriptRemotePath}", scriptPath)

    # Put the required test in the batch queue
    subprocessOutput = exec_remote_cmds(commands)

    # Retrieve process id
    out = subprocessOutput.stdout.split("\n")
    jobIds = [int(line.replace("Submitted batch job ", "")) for line in out if "Submitted batch job " in line]
    print("Job id : ", jobIds)

    # Wait until all process have terminated
    startingTime :int = time.time()
    while True:
        states = set()
        out = exec_remote_cmds(["squeue -h --format=\"%i %t %R %.10M\""])
        queue = out.stdout.split("\n")
        for process in queue:
            for jobId in jobIds:
                if str(jobId) in process:
                    states.add(process.split(" ")[1])
                    
        if len(states) > 0:
            currentTime :int = time.time()
            min = int((currentTime - startingTime) // 60)
            sec = int((currentTime - startingTime) - min*60)
            print(f"Process running... [{min}:{sec}] Current states :", str(states)[1:-1], end="----------\r")
            time.sleep(3)
        else:
            print()
            break


    ##############################################################
    ####################   RETRIEVE RESULTS   ####################
    ##############################################################

    # Retrieve the complete stdout and stderr log files path (only if there are non instrumented tests)
    outputFiles = []
    localFilePaths = []
    if jobs > 0:
        time.sleep(1)
        programsOutputs = exec_remote_cmds(["ls -1 {folder} | grep {scriptName}".format(folder=remoteOutputsFolder, scriptName=get_script_name(testId, dataset, False, version)[:-3])])
        outputFiles = programsOutputs.stdout.split("\n")[:-1] # Avoid last \n

        remoteFilePaths :str = [remoteOutputsFolder + "/" + outputFile for outputFile in outputFiles if not(instrument) or str(jobIds[instrumentedTestInternalId]) not in outputFile]
        localFilePaths :List[str] = [os.path.join(outFolder, outputFile) for outputFile in outputFiles  if not(instrument) or str(jobIds[instrumentedTestInternalId]) not in outputFile]


        # Copy .out and .err outputs from remote to local
        print("Downlading results ... ")
        for remoteFilePath in remoteFilePaths:
            transfer_from_remote(os.path.dirname(localFilePaths[0]), remoteFilePath)



    ##############################################################
    ######################   GLOBAL STATS   ######################
    ##############################################################

    # Check if some executions have failed
    exitSuccessCounter :int = 0
    exitFailureCounter :int = 0
    unknownCounter :int = 0
    filesToAnalyze :List[str] = []
    for localFilePath in [localFilePath for localFilePath in localFilePaths if localFilePath[-3:] == "err"]:
        with open(localFilePath, mode="r") as f:
            data = f.readlines()
            if len(data) > 0:
                lastLine :str = data[-1] 
            else :
                lastline :str = ""
            if "train_gpt2 : EXIT_SUCCESS" in lastLine:
                exitSuccessCounter += 1
                filesToAnalyze.append(localFilePath.replace(".err", ".out"))
            elif "train_gpt2 : EXIT_FAILURE" in lastLine:
                exitFailureCounter += 1
                print("Unsuccessful :", localFilePath)
            else :
                unknownCounter += 1
                print("Unknown :", localFilePath)
    if jobs > 0:
        print(f"\nTotal over {jobs} expected process :")
        print("\tEXIT_SUCCESS :", exitSuccessCounter)
        print("\tEXIT_FAILURE :", exitFailureCounter)
        print("\tEXIT_UNKNOWN :", unknownCounter)
        print(f"\nTOTAL : {exitSuccessCounter+exitFailureCounter+unknownCounter}/{jobs}\n")
        


    ##############################################################
    ########################   ANALYSIS   ########################
    ##############################################################





    #####################   COMPUTE TRACE   ######################

    if instrument:
        intrumentedJobId = jobIds[instrumentedTestInternalId]
        outputFileName = f"{remoteOutputsFolder}/{get_script_name(testId, dataset, True, version)[:-3]}-{intrumentedJobId}.err"
        transfer_from_remote(outFolder, outputFileName)
        outputFileName = f"{remoteOutputsFolder}/{get_script_name(testId, dataset, True, version)[:-3]}-{intrumentedJobId}.out"
        transfer_from_remote(outFolder, outputFileName)
        localOutputFileName = os.path.join(outFolder, f"{get_script_name(testId, dataset, True, version)[:-3]}-{intrumentedJobId}.err")
        isInstrumentationSuccess = False
        with open(localOutputFileName, mode="r") as f:
            data = f.readlines()
            if len(data) > 0:
                lastLine :str = data[-1] 
            else :
                lastline :str = ""
            if "train_gpt2 : EXIT_SUCCESS" in lastLine:
                print("Instrumentation Succesful")
                isInstrumentationSuccess = True
            elif "train_gpt2 : EXIT_FAILURE" in lastLine:
                print("Instrumentation Unsuccesful")
            else :
                print("Instrumentation unknown output")
        
        if isInstrumentationSuccess:
            out = exec_remote_cmds([f"cat {remoteTmpFolder}/node-{intrumentedJobId}"])
            instrumentedNodeName = str(out.stdout[:-1])
            
            instrumentationType = instrumentationTypes[version] if instrument else "None"
            traceFolder = traceFolders[version]
            match instrumentationType:
                case "extrae":
                    exec_remote_cmds(["module load extrae && ${EXTRAE_HOME}/bin/mpi2prv" + f" -sort-addresses -translate-addresses -trace-overwrite -e {remoteFolder}/{binaries[version]}-i -f {remoteTmpFolder}/{intrumentedJobId}/TRACE.mpits -o {remoteTracesFolder}/test-{testId}-extrae.prv"])
                    
                case "ovni":
                    ovniOutputName :str = f"loom.{instrumentedNodeName}."
                    
                    cmds = [
                        f"cd {remoteFolder}",
                        "module load ovni",
                        "module load ompss-2/git",
                        #"export NOSV_HOME=/home/bsc/bsc015728/instr/nos-v/install",
                        f"""if [ ! -d "{remoteTmpFolder}/{testId}" ]; then mkdir {remoteTmpFolder}/{testId}; fi""",
                        f"ls -1 ovni | grep {ovniOutputName}" + " | xargs -I '{}' mv ovni/{} " + f"{remoteTmpFolder}/{testId}/",
                        # f"sbatch -A bsc15 -q {queueName} {remoteMN5_scriptsFolder}/ovniemu.sh {remoteTmpFolder}/{testId}"
                        f"/home/bsc/bsc015728/instr/ovni/install/bin/ovniemu {remoteTmpFolder}/{testId}"
                        ]
                    out = exec_remote_cmds(cmds)
                    # startingTime :int = time.time()
                    # out = out.stdout.split("\n")
                    # ovniJobId = [int(line.replace("Submitted batch job ", "")) for line in out if "Submitted batch job " in line][0]
                    # As this job will take a few minutes, we will come to it later
                
        
    #####################   DATA ANALYSIS   #####################
    if len(filesToAnalyze) == 0:
        print("No succesful uninstrumented executions to analyse")
    else:
        print("Execution(s) analysis...")
        if jobs > 0:
            pythonScript :str = os.path.join(analysisFolder, "output_to_csv.py")
            if not(os.path.isfile(pythonScript)):
                print("python3 script", pythonScript, "not found")
                exit(1)

            csvFile = os.path.join(csvFolder, "-".join((os.path.basename(filesToAnalyze[0]).split("-")[:-1])) + ".csv")
            # Raw output to clean csv
            cmd = ["python3", pythonScript, csvFile]
            cmd.extend([path for path in filesToAnalyze if path[-3:] == 'out'])
            exec_local_cmds(cmd)

            pythonScript :str = os.path.join(analysisFolder, "csv_to_model_validity.py")
            exec_local_cmds(["python3", pythonScript, f"{csvFile}"], True)

            pythonScript :str = os.path.join(analysisFolder, "csv_to_graph.py")
            exec_local_cmds(["python3", pythonScript, f"{csvFile}", str(threads)], True)

            print("Done")

    #####################   CLEAN PARAVER   ######################

    if instrument and isInstrumentationSuccess:
        print("Waiting for paraver traces and cleaning up...")
        match(instrumentationType):
            case "ovni":
                # while True:
                #     states = set()
                #     out = exec_remote_cmds(["squeue -h --format=\"%i %t %R %.10M\""])
                #     queue = out.stdout.split("\n")
                #     for process in queue:
                #         if str(ovniJobId) in process:
                #             states.add(process.split(" ")[1])
                        
                #     if len(states) > 0:
                #         currentTime :int = time.time()
                #         min = int((currentTime - startingTime) // 60)
                #         sec = int((currentTime - startingTime) - min*60)
                #         print(f"Ovniemu running... [{min}:{sec}] Current states :", str(states)[1:-1], end="----------\r")
                #         time.sleep(10)
                #     else:
                #         print()
                #         break
                    
                cmds = [f"mv {remoteTmpFolder}/{testId}/cpu.prv {remoteTracesFolder}/test-{testId}-cpu.prv",
                f"mv {remoteTmpFolder}/{testId}/thread.prv {remoteTracesFolder}/test-{testId}-thread.prv",
                f"mv {remoteTmpFolder}/{testId}/cpu.pcf {remoteTracesFolder}/test-{testId}-cpu.pcf",
                f"mv {remoteTmpFolder}/{testId}/thread.pcf {remoteTracesFolder}/test-{testId}-thread.pcf",
                f"mv {remoteTmpFolder}/{testId}/cpu.row {remoteTracesFolder}/test-{testId}-cpu.row",
                f"mv {remoteTmpFolder}/{testId}/thread.row {remoteTracesFolder}/test-{testId}-thread.row"
                ]
                
                exec_remote_cmds(cmds)
                transfer_from_remote(traceFolder + "/", f"{remoteTracesFolder}/test-{testId}-thread.*")
                transfer_from_remote(traceFolder + "/", f"{remoteTracesFolder}/test-{testId}-cpu.*")
                exec_remote_cmds([f"rm -rf {remoteTmpFolder}/{testId}"])
                
            case "extrae":
                if downloadTrace:
                    print("Downloading paraver traces... This can take a few minutes...")
                    transfer_from_remote(traceFolder + "/", f"{remoteTracesFolder}/test-{testId}-extrae.prv")
                    transfer_from_remote(traceFolder + "/", f"{remoteTracesFolder}/test-{testId}-extrae.pcf")
                    transfer_from_remote(traceFolder + "/", f"{remoteTracesFolder}/test-{testId}-extrae.row")
                exec_remote_cmds([f"rm -rf {remoteTmpFolder}/{intrumentedJobId}"])
        exec_remote_cmds([f"rm -f {remoteTmpFolder}/node-{intrumentedJobId}"])
    else:
        print("Cleaning up...")
        
    exec_remote_cmds([f"rm -rf {remoteOutputsFolder}/test-{testId}*"])
        
    print("Done")
except Exception as e:
    print(e)
    clean_up(testId, jobIds, version)
