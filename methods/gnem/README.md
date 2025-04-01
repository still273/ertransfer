# GNEM

https://github.com/ChenRunjin/GNEM

## Apptainer

```bash
mkdir -p ../../apptainer ../../output/gnem
apptainer build ../../apptainer/gnem.sif container.def
srun -p ampere --gpus=1 apptainer run ../../apptainer/gnem.sif ../../datasets/d2_abt_buy/ ../../output/gnem/


# dev mode with bind
srun -p ampere --gpus=1 apptainer run --bind ./:/srv ../../apptainer/gnem.sif ../../datasets/d2_abt_buy/ ../../output/gnem/ -if v -tf -pt -le
# Application to other dataset:
srun -p ampere --gpus=1 apptainer run --bind ./:/srv ../../apptainer/gnem.sif ../../datasets/d2_abt_buy/ ../../output/gnem/ -if v -tf -pt -le -t ../../datasets/d3_amazon_google/kj_split/ 
```
