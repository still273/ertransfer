# Magellan

https://github.com/anhaidgroup/py_entitymatching
https://github.com/nishadi/py_entitymatching_sample

The docker image should be self-contained and should be able to run the method without any additional user input.
The image should contain the source code of the method and input + output transformers.

## Structure

- [Dockerfile](Dockerfile) should contain the instructions to build the docker image.
- [transform.py](transform.py) contains pre-processing and post-processing functions that will be applied to the input and output data.
- [entrypoint.py](entrypoint.py) should contain the main method that will be executed inside the docker container.

## How to use

You can directly execute the docker image as following:

```bash
docker run --rm -v .:/data magellan
```

This will assume that you have the input dataset in the current directory,
it will mount it as `/data` and will output the results in the `output` subdirectory.

You can override the input and output directories by providing them as arguments to the docker image:

```bash
docker run -v ../../datasets/d2_abt_buy:/data/input:ro -v ../../test:/data/output magellan /data/input /data/output
```

## Apptainer

```bash
mkdir -p ../../apptainer ../../output/magellan
apptainer build ../../apptainer/magellan.sif container.def
srun --gpus=1 apptainer run ../../apptainer/magellan.sif ../../datasets/d2_abt_buy/ ../../output/magellan/

# dev mode with bind
srun --gpus=1 apptainer run --bind ./:/srv ../../apptainer/magellan.sif ../../datasets/d2_abt_buy/ ../../output/magellan/
```

## Latest error

```bash
Attribute (tableA_id ) does not qualify  to be a key; Not setting/replacing the key
Attribute (tableA_id ) does not qualify  to be a key; Not setting/replacing the key
Requested metadata ( key ) for the given DataFrame is not present in the catalog
Traceback (most recent call last):
  File "/srv/entrypoint.py", line 80, in <module>
    train_f_vectors = em.extract_feature_vecs(train, feature_table=F, attrs_after='label')
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/py_entitymatching/feature/extractfeatures.py", line 203, in extract_feature_vecs
    cm.get_metadata_for_candset(
  File "/usr/local/lib/python3.11/site-packages/py_entitymatching/catalog/catalog_manager.py", line 1293, in get_metadata_for_candset
    key = get_key(candset)
          ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/py_entitymatching/catalog/catalog_manager.py", line 661, in get_key
    return get_property(data_frame, 'key')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/py_entitymatching/catalog/catalog_manager.py", line 71, in get_property
    raise KeyError(
KeyError: 'Requested metadata ( key ) for the given DataFrame is not present in the catalog'
```