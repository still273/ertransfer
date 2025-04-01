# DITTO

https://github.com/megagonlabs/ditto

entrypoint.py based on train_ditto.py and matcher.py


## Apptainer

```bash
mkdir -p ../../apptainer ../../output/ditto
apptainer build ../../apptainer/ditto.sif container.def
srun -p ampere --gpus=1 apptainer run ../../apptainer/ditto.sif ../../datasets/d2_abt_buy/ ../../output/ditto/

# dev mode with bind
srun -p ampere --gpus=1 apptainer run --bind ./:/srv ../../apptainer/ditto.sif ../../datasets/d2_abt_buy/kj_split ../../output/ditto/ -if v -pt -le

#apply to different dataset:
srun -p ampere --gpus=1 apptainer run --bind ./:/srv ../../apptainer/ditto.sif ../../datasets/d2_abt_buy/kj_split ../../output/ditto/ -if v -tf -pt -le -t ../../datasets/d3_amazon_google/kj_split/ 
```

