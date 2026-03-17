#!/bin/ksh
#$ -q gpu
#$ -o result.out
#$ -j y
#$ -N ia2hl_foundation
#$ -cwd

cd $WORKDIR
source /beegfs/data/work/imvia/in156281/H-optimus-0-valis-lora/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/H-optimus-0-valis-lora/venv/lib/python3.9/site-packages:$PYTHONPATH
export TORCH_HOME=/beegfs/data/work/imvia/in156281/H-optimus-0-valis-lora/torch_cache
export HF_HOME=/beegfs/data/work/imvia/in156281/H-optimus-0-valis-lora/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
cd $WORKDIR/H-optimus-0-valis-lora
python train.py --data_dir dataset --output_dir checkpoints --epochs 20 --batch_size 2 --num_workers 2 --max_train_batches 100 --max_valid_batches 100