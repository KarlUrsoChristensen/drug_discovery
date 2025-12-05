#!/bin/bash
#BSUB -J mat
#BSUB -o /dtu/blackhole/0a/224426/logs/run_%J.out
#BSUB -e /dtu/blackhole/0a/224426/logs/run_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 4:00
#BSUB -q gpua100 
#BSUB -gpu "num=1:mode=exclusive_process"

export TMPDIR="/dtu/blackhole/0a/224426/tmp"
export HYDRA_FULL_ERROR=1

source /zhome/e2/6/224426/project/drug_discovery/.venv/bin/activate
cd /zhome/e2/6/224426/project/drug_discovery/src

# Uses model specified in run.yaml
python run.py --config-name=run \
  hydra.run.dir=/dtu/blackhole/0a/224426/runs/original \
  result_dir=/dtu/blackhole/0a/224426/runs/original_norm_skip_drop

echo "✅ Training completed!"

