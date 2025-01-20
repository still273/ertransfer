# EMTransformer

https://github.com/gpapadis/DLMatchers/tree/main/EMTransformer
entrypoint.py based on run_all.py

## How to use

You can directly execute the docker image as following:

```bash
docker run --rm -v .:/data emtransformer
```

This will assume that you have the input dataset in the current directory,
it will mount it as `/data` and will output the results in the `output` subdirectory.

You can override the input and output directories by providing them as arguments to the docker image:

```bash
docker run -v ../../datasets/d2_abt_buy:/data/input:ro -v ../../test:/data/output emtransformer /data/input /data/output
```

## Latest error

```bash
2024-07-17T12:51:13.177118254Z Traceback (most recent call last):
2024-07-17T12:51:13.177123935Z   File "./entrypoint.py", line 41, in <module>
2024-07-17T12:51:13.177318428Z     train_df, test_df = transform_input(args.input, columns_to_join, ' ', [prefix_1, prefix_2])
2024-07-17T12:51:13.177333538Z   File "/workspace/transform.py", line 34, in transform_input
2024-07-17T12:51:13.177340652Z     train = join_columns(train_df, columns_to_join, separator, prefixes)
2024-07-17T12:51:13.177346454Z   File "/workspace/transform.py", line 8, in join_columns
2024-07-17T12:51:13.177492973Z     agg_table = table['id']
2024-07-17T12:51:13.177504075Z   File "/opt/conda/lib/python3.7/site-packages/pandas/core/frame.py", line 3458, in __getitem__
2024-07-17T12:51:13.178376728Z     indexer = self.columns.get_loc(key)
2024-07-17T12:51:13.178405294Z   File "/opt/conda/lib/python3.7/site-packages/pandas/core/indexes/base.py", line 3363, in get_loc
2024-07-17T12:51:13.179940585Z     raise KeyError(key) from err
2024-07-17T12:51:13.180101232Z KeyError: 'id'
```
