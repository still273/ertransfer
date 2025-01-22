# ZeroER

https://github.com/nishadi/zeroer
entrypoint.py based on zeroer.py

## How to use

You can directly execute the docker image as following:

```bash
docker run --rm -v .:/data zeroer
```

This will assume that you have the input dataset in the current directory,
it will mount it as `/data` and will output the results in the `output` subdirectory.

You can override the input and output directories by providing them as arguments to the docker image:

```bash
docker run -v ../../datasets/d2_abt_buy:/data/input:ro -v ../../test:/data/output zeroer /data/input /data/output
```

## Apptainer

```bash
mkdir -p ../../apptainer ../../output/zeroer
apptainer build ../../apptainer/zeroer.sif container.def
srun --gpus=1 apptainer run ../../apptainer/zeroer.sif ../../datasets/d2_abt_buy/ ../../output/zeroer/

# dev mode with bind
srun --gpus=1 apptainer run --bind ./:/srv ../../apptainer/zeroer.sif ../../datasets/d2_abt_buy/ ../../output/zeroer/
```
