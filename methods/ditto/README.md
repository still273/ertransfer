# DITTO

https://github.com/nishadi/ditto

entrypoint.py based on train_ditto.py and matcher.py

## How to use

You can directly execute the docker image as following:

```bash
docker run --rm -v .:/data ditto
```

This will assume that you have the input dataset in the current directory,
it will mount it as `/data` and will output the results in the `output` subdirectory.

You can override the input and output directories by providing them as arguments to the docker image:

```bash
docker run -v ../../datasets/d2_abt_buy:/data/input:ro -v ../../test:/data/output ditto /data/input /data/output
```

## Apptainer

```bash
mkdir -p ../../apptainer ../../output/ditto
apptainer build ../../apptainer/ditto.sif container.def
srun --gpus=1 apptainer run ../../apptainer/ditto.sif ../../datasets/d2_abt_buy/ ../../output/ditto/

# dev mode with bind
srun --gpus=1 apptainer run --bind ./:/srv ../../apptainer/ditto.sif ../../datasets/d2_abt_buy/ ../../output/ditto/
```

## Last error

```bash
Input directory:  ['tableA.csv', 'test.csv', 'tableB.csv', 'train.csv', 'matches.csv']
Traceback (most recent call last):
  File "/srv/entrypoint.py", line 45, in <module>
    print("Output directory: ", os.listdir(args.output))
FileNotFoundError: [Errno 2] No such file or directory: 'output'
srun: error: gpunode06: task 0: Exited with exit code 1
```
