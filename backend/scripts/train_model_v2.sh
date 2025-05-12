#!/usr/bin/env bash

JOB=$(date +%Y%m%d%H%M%S)

echo "device_ids: '0'" >> ${JOB}.yaml
echo "device_num: 1" >> ${JOB}.yaml
echo "max_epochs: 50" >> ${JOB}.yaml
echo "save_path: ${EXPERIMENTS_DIR}/${JOB}" >> ${JOB}.yaml
echo "check_point_name: model${JOB}" >> ${JOB}.yaml

echo "DataLoader:" >> ${JOB}.yaml
echo "  batch_size: 32" >> ${JOB}.yaml
echo "  num_workers: 8" >> ${JOB}.yaml

echo "model:" >> ${JOB}.yaml # TODO update the following parameters
echo "  in_feature_dim: 3" >> ${JOB}.yaml
echo "  feature_dim: 256" >> ${JOB}.yaml
echo "  time_steps: 121" >> ${JOB}.yaml
echo "  head_num: 4" >> ${JOB}.yaml
echo "  k: 4" >> ${JOB}.yaml
echo "  F: 6" >> ${JOB}.yaml
echo "  lr: 1e-5" >> ${JOB}.yaml

# Save experiment settings.
mkdir -p ${EXPERIMENTS_DIR}/${JOB}
mv ${JOB}.yaml ${EXPERIMENTS_DIR}/${JOB}/

cd ${PROJECT_DIR}
nohup uv run python train_v2.py ${JOB} > ${EXPERIMENTS_DIR}/${JOB}/train.log &