
#!/bin/bash
#BSUB -J mathilde_sweep
#BSUB -o /dtu/blackhole/0a/224426/logs/sweep_%J.out
#BSUB -e /dtu/blackhole/0a/224426/logs/sweep_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"

# Sæt environment variables
export TMPDIR="/dtu/blackhole/0a/224426/tmp"
export HYDRA_FULL_ERROR=1

# Aktiver environment
source /dtu/blackhole/0a/224426/drug_discovery_env/bin/activate

# Gå til src directory
cd /zhome/e2/6/224426/project/drug_discovery/src

# Kør sweep med alle outputs til blackhole
python run.py --config-name=sweep --multirun \
