# Embeddings with BERT or RoBERTa


## Apptainer

WARNING: This method doesn't use GPUs.

```bash
mkdir -p ../../apptainer ../../output/embeddings
apptainer build ../../apptainer/embeddings.sif container.def
srun -p ampere --gpus=1 apptainer run ../../apptainer/embeddings.sif ../../datasets/d2_abt_buy/ ../../output/embeddings/

# dev mode with bind
srun -p ampere --gpus=1 apptainer run --bind ./:/srv ../../apptainer/embeddings.sif ../../datasets/d2_abt_buy/ ../../output/embeddings/
```
