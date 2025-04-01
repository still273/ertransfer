# EMTransformer
https://github.com/brunnurs/entity-matching-transformer

## Apptainer

```bash
mkdir -p ../../apptainer ../../output/emtransformer
apptainer build ../../apptainer/emtransformer.sif container.def
srun -p ampere --gpus=1 apptainer run ../../apptainer/emtransformer.sif ../../datasets/d2_abt_buy/ ../../output/emtransformer/

# dev mode with bind
srun -p ampere --gpus=1 apptainer run --bind ./:/srv ../../apptainer/emtransformer.sif ../../datasets/d2_abt_buy/ ../../output/emtransformer/ -if v -tf -pt -le
#apply to different dataset
srun -p ampere --gpus=1 apptainer run --bind ./:/srv ../../apptainer/emtransformer.sif ../../datasets/d2_abt_buy/ ../../output/emtransformer/ -if v -tf -pt -le -t ../../datasets/d3_amazon_google/kj_split/ 
```
