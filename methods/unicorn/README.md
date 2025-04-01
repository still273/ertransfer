# DADER

## Apptainer

```bash
mkdir -p ../../apptainer ../../output/unicorn
apptainer build ../../apptainer/unicorn.sif container.def
srun -p ampere --gpus=1 apptainer run ../../apptainer/unicorn.sif ../../datasets/d2_abt_buy/ ../../output/ditto/

# dev mode with bind
srun -p ampere --gpus=1 apptainer run --bind ./:/srv ../../apptainer/unicorn.sif ../../datasets/d2_abt_buy/kj_split ../../output/dader/ -if vt -pt -le
```
