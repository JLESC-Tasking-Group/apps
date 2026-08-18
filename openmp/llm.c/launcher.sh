#!/bin/bash

usage() {
    echo "Usage: $0 [-v <sequential|openmp|openmpv|ompss|ompss-mpi|openmpv-mpi>] [-i] [-m <integer>] [-c <integer>] [-n <integer>] [-A <Account>] [-q <queue>]"
    echo '-v : Version to execute'
    echo '-i : Enable instrumentation'
    echo '-m : Set nosV shared_memory.size' in Gigabytes
    echo '-c : Number of CPUs per rank or process'
    echo '-n : Number of ranks (MPI only)'
    echo '-A : Slurm account (MPI only)'
    echo '-q : Slurm queue (MPI only)'
    exit 1
}

# Default values
version=""
instrument=false # Default to false
nosv_shared_mem=""
num_cores=1
ranks="1"
instrument_vars=""
slurm_account=""
slurm_queue=""
mpi_enabled="false"

# Parse options
while getopts ":v:im:c:n:A:q:h" opt; do
    case "${opt}" in
    v)
        version="${OPTARG}"
        if [[ "$version" == 'ompss-mpi' || "$version" == 'oss-mpi' || "$version" == 'openmpv-mpi' || "$version" == 'ompv-mpi' ]]; then
        mpi_enabled=true
        fi
        ;;
    i)
        instrument=true
        ;;
    m) 
        nosv_shared_mem="NOSV_CONFIG_OVERRIDE=shared_memory.size=${OPTARG}G"
        ;;
    c)
        if [[ "${OPTARG}" =~ ^[0-9]+$ ]]; then
            num_cores="${OPTARG}"
        else
            echo "Error: -c must be an integer."
            usage
        fi
        ;;
    n)
        if [[ "${OPTARG}" =~ ^[0-9]+$ ]]; then
            ranks="${OPTARG}"
        else
            echo "Error: -n must be an integer."
            usage
        fi
        ;;
    A)
        slurm_account="-A ${OPTARG}"
        ;;
    q)
        slurm_queue="-q ${OPTARG}"
        ;;
    h)
        usage
        ;;
    \?)
        echo "Unknown option: -${OPTARG}"
        usage
        ;;
    :)
        echo "Option -${OPTARG} requires an argument."
        usage
        ;;
    esac
done

# Ensure version is specified
if [[ -z "$version" ]]; then
    echo "Error: -v <version> is required."
    usage
fi

# Ensure ranks is provided if version is mpi or tampi
if [[ $mpi_enabled && -z "$ranks" ]]; then
    echo "Error: -n <ranks> is required when version is using mpi."
    usage
fi
if [[ ! "$mpi_enabled" && "$ranks" -gt 1 ]]; then
    echo "Error: Can't create more than one rank for non-distributed versions."
    usage
fi

# Check instrument compatibility and set instrument_vars
if [[ "$instrument" == "true" ]]; then
    case "$version" in
    "sequential" | "s")
        echo "Error: Instrumentation is incompatible with 'sequential' version."
        exit 1
        ;;
    "openmp" | "omp")
        echo "Error: Instrumentation is incompatible with 'openmp'. Use 'openmpv' instead."
        exit 1
        ;;
    "openmpv" | "ompv")
        instrument_vars='OMP_OVNI=2 NOSV_CONFIG_OVERRIDE=instrumentation.version=ovni'
        ;;
    "ompss" | "oss")
        instrument_vars='NODES_OVNI=1 NOSV_CONFIG_OVERRIDE=instrumentation.version=ovni'
        ;;
    "ompss-mpi" | "oss-mpi")
        instrument_vars='NOSV_CONFIG_OVERRIDE=instrumentation.version=ovni TAMPI_INSTRUMENT=ovni'
        ;;
    "openmp-mpi" | "omp-mpi")
        instrument_vars='OMP_OVNI=2 NOSV_CONFIG_OVERRIDE=instrumentation.version=ovni TAMPI_INSTRUMENT=ovni'
        ;;
    *)
        echo "Error: Unsupported version for instrumentation."
        usage
        ;;
    esac
fi

echo
echo "Version : $version"
echo "MPI enabled : $mpi_enabled"
echo "Instrument : $instrument"
echo "CPU per rank : $num_cores"
echo -e "Ranks : $ranks \n\n"

# Based on version, execute the appropriate command
if [[ "$version" == "sequential" || "$version" == "s" ]]; then
    echo -e '$> ./train_gpt2\n'
    ./train_gpt2
elif [[ "$version" == "openmp" || "$version" == "omp" ]]; then
    echo -e "$> OMP_NUM_THREADS=$num_cores ./train_gpt2_openmp \n"
    env OMP_NUM_THREADS=$num_cores ./train_gpt2_openmp
elif [[ "$version" == "openmpv" || "$version" == "ompv" ]]; then
    echo -e "$> OMP_NUM_THREADS=$num_cores $instrument_vars ./train_gpt2_openmpv \n"
    env OMP_NUM_THREADS=$num_cores $instrument_vars ./train_gpt2_openmpv
elif [[ "$version" == "ompss" || "$version" == "oss" ]]; then
    echo -e "$> $nosv_shared_mem OMP_NUM_THREADS=$num_cores $instrument_vars taskset -c 0-$(($num_cores - 1)) ./train_gpt2_ompss \n"
    env $nosv_shared_mem OMP_NUM_THREADS=$num_cores $instrument_vars taskset -c 0-$(($num_cores - 1)) ./train_gpt2_ompss
elif [[ "$version" == "ompss-mpi" || "$version" == "ompss-mpi" ]]; then
    echo -e "$> $nosv_shared_mem TAMPI_POLLING_PERIOD=0 NOSV_PRESET=isolated OMP_NUM_THREADS=$num_cores $instrument_vars srun $slurm_account $slurm_queue -n $ranks -c $num_cores ./train_gpt2_ompss-mpi \n"
    env $nosv_shared_mem TAMPI_POLLING_PERIOD=0 NOSV_PRESET=isolated OMP_NUM_THREADS=$num_cores $instrument_vars srun $slurm_account $slurm_queue -n $ranks -c $num_cores ./train_gpt2_ompss-mpi
elif [[ "$version" == "openmpv-mpi" || "$version" == "ompv-mpi" ]]; then
    echo -e "$> TAMPI_POLLING_PERIOD=0 NOSV_PRESET=isolated OMP_NUM_THREADS=$num_cores $instrument_vars srun $slurm_account $slurm_queue -n $ranks -c $num_cores ./train_gpt2_openmpv-mpi \n"
    env TAMPI_POLLING_PERIOD=0 NOSV_PRESET=isolated OMP_NUM_THREADS=$num_cores $instrument_vars srun $slurm_account $slurm_queue -n $ranks -c $num_cores ./train_gpt2_openmpv-mpi
else
    echo "Error: Unsupported version specified."
    usage
fi
