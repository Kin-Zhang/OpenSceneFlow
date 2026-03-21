#!/bin/bash
#SBATCH -J synflow
#SBATCH --mem 500GB
#SBATCH --gres gpu:10
#SBATCH --cpus-per-task 48
#SBATCH --constrain "galadriel|eowyn"
#SBATCH --output /Midgard/home/qingwen/logs/synflow/%J.out
#SBATCH --error  /Midgard/home/qingwen/logs/synflow/%J.err

PYTHON=/Midgard/home/qingwen/miniforge3/envs/seflow/bin/python
cd /Midgard/home/qingwen/workspace/OpenSceneFlow

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Midgard/home/qingwen/miniforge3/lib

DATA_DIR="/Midgard/datasets/opensf"
# 4k dataset
$PYTHON train.py slurm_id=$SLURM_JOB_ID wandb_mode=online wandb_project_name=synflow \
     train_data="['$DATA_DIR/data-32-160k-1k', '$DATA_DIR/data-32-160k-2k', '$DATA_DIR/data-64-600k-6k', '$DATA_DIR/data-64-460k-7k']" \
     val_data='/local_storage/datasets/qingwen/data/h5py/av2/val' model=deltaflow loss_fn=deltaflowLoss model.target.decoder_option=default \
     num_workers=16 num_frames=5 model.target.decay_factor=0.4 epochs=21 batch_size=2 \
     save_top_model=3 val_every=3 train_aug=True "voxel_size=[0.15, 0.15, 0.15]" "point_cloud_range=[-38.4, -38.4, -3, 38.4, 38.4, 3]" \
     optimizer.lr=2e-4 +optimizer.scheduler.name=StepLR +optimizer.scheduler.step_size=3 +optimizer.scheduler.gamma=0.9
