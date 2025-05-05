#!/usr/bin/env bash

JOB=$(date +%Y%m%d%H%M%S)

echo "train:" >> ${JOB}.yaml
# task=basketball  # "basketball" or "toy".
# echo "  task: ${task}" >> ${JOB}.yaml

echo "  train_samples_per_epoch: 20000" >> ${JOB}.yaml
echo "  valid_samples: 1000" >> ${JOB}.yaml
echo "  workers: 10" >> ${JOB}.yaml
echo "  learning_rate: 1.0e-5" >> ${JOB}.yaml
echo "  patience: 20" >> ${JOB}.yaml
echo "  max_epochs: 1000000" >> ${JOB}.yaml
echo "  beta: 1" >> ${JOB}.yaml

echo "dataset:" >> ${JOB}.yaml
echo "  hz: 5" >> ${JOB}.yaml
echo "  secs: 6" >> ${JOB}.yaml
echo "  player_traj_n: 10" >> ${JOB}.yaml
echo "  max_player_move: 4.5" >> ${JOB}.yaml

echo "model:" >> ${JOB}.yaml # TODO update the following parameters
echo "  input_dim: 50" >> ${JOB}.yaml
echo "  hidden_dim: 256" >> ${JOB}.yaml
echo "  latent_dim: 64" >> ${JOB}.yaml
echo "  n_heads: 8" >> ${JOB}.yaml
echo "  n_layers: 3" >> ${JOB}.yaml


# Save experiment settings.
mkdir -p ${EXPERIMENTS_DIR}/${JOB}
mv ${JOB}.yaml ${EXPERIMENTS_DIR}/${JOB}/

gpu=0
cd ${PROJECT_DIR}
nohup uv run python train.py ${JOB} ${gpu} > ${EXPERIMENTS_DIR}/${JOB}/train.log &