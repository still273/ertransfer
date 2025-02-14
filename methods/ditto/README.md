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
Traceback (most recent call last):
  File "/srv/entrypoint.py", line 56, in <module>
    trainset, testset, train_ids, test_ids = transform_input_old(args.input, temp_output, args.recall, seed=args.seed)
  File "/srv/transform.py", line 121, in transform_input_old
    pairs = generate_candidates(tableA_df, tableB_df, matches_df, seed)
  File "/srv/transform.py", line 21, in generate_candidates
    (cand_tableB.iloc[matches_df['tableB_id']]).reset_index(drop=True)
  File "/opt/conda/lib/python3.7/site-packages/pandas/core/indexing.py", line 931, in __getitem__
    return self._getitem_axis(maybe_callable, axis=axis)
  File "/opt/conda/lib/python3.7/site-packages/pandas/core/indexing.py", line 1557, in _getitem_axis
    return self._get_list_axis(key, axis=axis)
  File "/opt/conda/lib/python3.7/site-packages/pandas/core/indexing.py", line 1533, in _get_list_axis
    raise IndexError("positional indexers are out-of-bounds") from err
IndexError: positional indexers are out-of-bounds
```
