#!/bin/bash

usage() {
    echo "Usage: $0 [-v <sequential|openmp|openmpv|ompss|ompss-mpi|ompss-openmpv>] [-i <integer>] [-B <integer>] [-T <integer>] [-t] [-d <tinyshakespeare|tinystories>] [-V] [-I]"
    echo '-v : Version to compile'
    echo '-i : Number of training iterations'
    echo '-B : Batch size'
    echo '-T : Token sequence size'
    echo '-t : Enable taskiter'
    echo '-d : Dataset'
    echo '-V : Disable model validation'
    echo '-I : Disable model inference'
    exit 1
}

# Default values
iterations=40
batch_size=4
sequence_size=1024
taskiter=false
dataset="tinystories"
skip_model_validation=false
skip_model_inference=false
version=""

# Parse options
while getopts ":v:i:B:T:td:VI" opt; do
    case "${opt}" in
    v)
        version="${OPTARG}"
        ;;
    i)
        if [[ "${OPTARG}" =~ ^[0-9]+$ && "${OPTARG}" -ge 1 ]]; then
            iterations="${OPTARG}"
        else
            echo "Error: -i must be an integer >= 1."
            usage
        fi
        ;;
    B)
        if [[ "${OPTARG}" =~ ^[0-9]+$ && "${OPTARG}" -ge 1 ]]; then
            batch_size="${OPTARG}"
        else
            echo "Error: -B must be an integer >= 1."
            usage
        fi
        ;;
    T)
        if [[ ("${OPTARG}" =~ ^[0-9]+$) && ("${OPTARG}" -ge 1) ]]; then
            sequence_size="${OPTARG}"
        else
            echo "Error: -T must be an integer >= 1."
            usage
        fi
        ;;
    t)
        taskiter=true
        ;;
    d)
        if [[ "${OPTARG}" == "tinyshakespeare" || "${OPTARG}" == "tinystories" ]]; then
            dataset="${OPTARG}"
        else
            echo "Error: -d must be either 'tinyshakespeare' or 'tinystories'."
            usage
        fi
        ;;
    V)
        skip_model_validation=true
        ;;
    I)
        skip_model_inference=true
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

echo
echo '----------------------------------------'
echo "Version (v): $version"
echo "training iterations (i) : $iterations"
echo "B (B) : $batch_size"
echo "T (T) : $sequence_size"
echo "Enable taskiter (t) : $taskiter"
echo "Dataset (d) : $dataset"
echo "Skip model validation (V) : $skip_model_validation"
echo "Skip model inference (I) : $skip_model_inference "
echo -e '---------------------------------------- \n\n'

# Create the VARS string with the provided settings
VARS="ITERATIONS=$iterations BATCH_SIZE=$batch_size SEQUENCE_SIZE=$sequence_size TASKITER=$taskiter DATASET=$(echo "$dataset" | tr '[:lower:]' '[:upper:]') SKIP_MODEL_VALIDATION=$skip_model_validation SKIP_MODEL_INFERENCE=$skip_model_inference"

# Based on version, execute the appropriate make command
if [[ "$version" == "sequential" || "$version" == "s" ]]; then
    echo -e '$> make train_gpt2 \n'
    make train_gpt2
elif [[ "$version" == "openmp" || "$version" == "omp" ]]; then
    echo -e "$> $VARS make train_gpt2_openmp \n"
    env $VARS make train_gpt2_openmp
elif [[ "$version" == "openmpv" || "$version" == "ompv" ]]; then
    echo -e "$> $VARS make train_gpt2_openmpv \n"
    env $VARS make train_gpt2_openmpv
elif [[ "$version" == "ompss" || "$version" == "oss" ]]; then
    echo -e "$> $VARS make train_gpt2_ompss \n"
    env $VARS make train_gpt2_ompss
elif [[ "$version" == "ompss-mpi" || "$version" == "oss-mpi" ]]; then
    echo -e "$> $VARS make train_gpt2_ompss-mpi \n"
    env $VARS make train_gpt2_ompss-mpi
elif [[ "$version" == "openmpv-mpi" || "$version" == "ompv-mpi" ]]; then
    echo -e "$> $VARS make train_gpt2_openmpv-mpi \n"
    env $VARS make train_gpt2_openmpv-mpi
else
    echo "Error: Unsupported version specified."
    usage
fi
