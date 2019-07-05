#!/usr/bin/env bash

config=$1
log=${DIR_LOGS}"/grid_iradon"

if [[ $# != 1 ]]; then
    echo "1 argument required: json config file path"
    exit 1
fi

echo "Preprocessing Sinogram"

${PYTHON_PATH} ${SCRIPTS_PATH}/splineradon/process_sinogram.py ${config}

if [[ "$?" != "0" ]]; then
    echo "Preprocessing failed!"
    exit 1
fi

echo "Filtering Sinogram"

n_jobs=10
queue.pl -l q_1day -N preprocess JOB=1:${n_jobs} ${log}/preprocess_JOB.log\
          ${PYTHON_PATH} ${SCRIPTS_PATH}/splineradon/iradon_preprocess.py -n ${n_jobs} -j JOB ${config}

if [[ "$?" != "0" ]]; then
    echo "Filtering sinogram failed!"
    exit 1
fi

${PYTHON_PATH} ${SCRIPTS_PATH}/splineradon/iradon_preprocess.py -n ${n_jobs} --merge ${config}

if [[ "$?" != "0" ]]; then
    echo "Merging sinogram failed!"
    exit 1
fi

echo "Running inverse radon"
n_jobs=45
queue.pl -l q_1day -N iradon -l h_vmem=8G JOB=1:${n_jobs} ${log}/iradon_JOB.log\
    ${PYTHON_PATH} ${SCRIPTS_PATH}/splineradon/distributed_iradon.py -n ${n_jobs} -j JOB ${config}

if [[ "$?" != "0" ]]; then
    echo "Reconstruction failed!"
    exit 1
fi

${PYTHON_PATH} ${SCRIPTS_PATH}/splineradon/distributed_iradon.py -n ${n_jobs} --merge ${config}

if [[ "$?" != "0" ]]; then
    echo "Merging reconstruction failed!"
    exit 1
fi
