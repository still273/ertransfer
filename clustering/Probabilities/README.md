# Probability Clustering


## Expected directory structure


## Apptainer

```bash
mkdir -p ../../apptainer ../../output
apptainer build ../../apptainer/prob_cluster.sif container.def
srun --gpus=0 -p ampere apptainer run ../../apptainer/prob_cluster.sif ../../output/ditto/d2_abt_buy/predictions_d3_amazon_google.csv

# dev mode with bind
srun --gpus=0 -p ampere apptainer run --bind ./:/srv ../../apptainer/prob_cluster.sif ../../output/ditto/d2_abt_buy/predictions_d3_amazon_google.csv
```
