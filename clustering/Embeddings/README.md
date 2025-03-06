# Probability Clustering

Splits the dataset into training, validation, and testing sets.
Based on [KNN-Join](https://pyjedai.readthedocs.io/en/latest/tutorials/SimilarityJoins.html).

## Expected directory structure


## Apptainer

```bash
mkdir -p ../../apptainer ../../output
apptainer build ../../apptainer/emb_cluster.sif container.def
srun --gpus=0 -p ampere apptainer run ../../apptainer/emb_cluster.sif ../../output/ditto/d2_abt_buy/predictions_d3_amazon_google.csv

# dev mode with bind
srun --gpus=0 -p ampere apptainer run --bind ./:/srv ../../apptainer/emb_cluster.sif ../../output/ditto/d2_abt_buy/predictions_d3_amazon_google.csv
```
