# DADER

## Apptainer

```bash
mkdir -p ../../apptainer ../../output/dader
apptainer build ../../apptainer/dader.sif container.def
srun -p ampere --gpus=1 apptainer run ../../apptainer/dader.sif ../../datasets/d2_abt_buy/ ../../output/ditto/

# dev mode with bind
srun -p ampere --gpus=1 apptainer run --bind ./:/srv ../../apptainer/dader.sif ../../datasets/d2_abt_buy/kj_split ../../output/dader/ -if vt -pt -le
```
