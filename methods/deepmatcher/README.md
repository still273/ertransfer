# DeepMatcher

https://github.com/anhaidgroup/deepmatcher

## How to use

IMPORTANT! `/workspace/embedding` should be mounted with `wiki.en.bin` embeddings inside.

You can directly execute the docker image as following:

```bash
docker run --rm -v .:/data deepmatcher
```

This will assume that you have the input dataset in the current directory,
it will mount it as `/data` and will output the results in the `output` subdirectory.

You can override the input and output directories by providing them as arguments to the docker image:

```bash
docker run -v ../../datasets/d2_abt_buy:/data/input:ro -v ../../test:/data/output -v ../../embedding:/workspace/embedding deepmatcher /data/input /data/output
```

## Apptainer

```bash
mkdir -p ../../apptainer ../../output/deepmatcher
apptainer build ../../apptainer/deepmatcher.sif container.def
srun --gpus=1 apptainer run ../../apptainer/deepmatcher.sif ../../datasets/d2_abt_buy/ ../../output/deepmatcher/ ../../embedding/
```

## Latest error

```bash
Traceback (most recent call last):
  File "/srv/entrypoint.py", line 63, in <module>
    transform_output(stats, test_time, test_max_mem, args.output)
  File "/srv/transform.py", line 12, in transform_output
    'time': [runtime],
  File "/opt/conda/lib/python3.7/site-packages/pandas/core/frame.py", line 614, in __init__
    mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
  File "/opt/conda/lib/python3.7/site-packages/pandas/core/internals/construction.py", line 465, in dict_to_mgr
    arrays, data_names, index, columns, dtype=dtype, typ=typ, consolidate=copy
  File "/opt/conda/lib/python3.7/site-packages/pandas/core/internals/construction.py", line 124, in arrays_to_mgr
    arrays = _homogenize(arrays, index, dtype)
  File "/opt/conda/lib/python3.7/site-packages/pandas/core/internals/construction.py", line 590, in _homogenize
    val, index, dtype=dtype, copy=False, raise_cast_failure=False
  File "/opt/conda/lib/python3.7/site-packages/pandas/core/construction.py", line 571, in sanitize_array
    subarr = maybe_convert_platform(data)
  File "/opt/conda/lib/python3.7/site-packages/pandas/core/dtypes/cast.py", line 118, in maybe_convert_platform
    arr = construct_1d_object_array_from_listlike(values)
  File "/opt/conda/lib/python3.7/site-packages/pandas/core/dtypes/cast.py", line 1990, in construct_1d_object_array_from_listlike
    result[:] = values
  File "/opt/conda/lib/python3.7/site-packages/torch/_tensor.py", line 680, in __array__
    return self.numpy().astype(dtype, copy=False)
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
srun: error: gpunode03: task 0: Exited with exit code 1
```
