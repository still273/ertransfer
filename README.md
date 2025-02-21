# No-code Benchmarking of Entity Resolution

https://swimlanes.io/u/k3Rmy375P

The main requirement is to implement the following process:

- to set several DL-based matching algorithms running on a server as Docker containers
- to feed one of them with three sets of record pairs from pyJedAI' blocking (training, validation and testing)
- to receive the labels of the pairs in the testing set

## Repository setup

```bash
git clone git@github.com:erbench/erbench.git
git submodule update --recursive --init
```

### Download embeddings

```bash
mkdir embedding
wget https://zenodo.org/record/6466387/files/wiki.en.bin -O embedding/wiki.en.bin
```

## Methods

| Name                                             | Container  | Input params                                                                      | Metrics columns                                     | Predictions columns                   |
| ------------------------------------------------ | ---------- | --------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------- |
| [splitter-simple](splitter-simple/README.md)     | ok, no GPU |                                                                                   |                                                     |                                       |
| [splitter](splitter/README.md)                   | ok         |                                                                                   |                                                     |                                       |
| [deepmatcher](methods/deepmatcher/README.md)     | ok         | input,output,embedding,--epochs                                                   | f1,precision,recall,train_time,eval_time            | tableA_id,tableB_id,label,prob_class1 |
| [ditto](methods/ditto/README.md)                 | fails      | input,output,--recall,--seed,--run_id,--model,--epochs                            |                                                     |                                       |
| [emtransformer](methods/emtransformer/README.md) | ok         | input,output,--recall,--seed,--model,--max_seq_length,--train_batch_size,--epochs | f1,precision,recall,train_time,eval_time (0-scores) | tableA_id,tableB_id (empty)           |
| [gnem](methods/gnem/README.md)                   | ok         | input,output,--epochs                                                             | f1,precision,recall,train_time,eval_time            | tableA_id,tableB_id                   |
| [hiermatcher](methods/hiermatcher/README.md)     | ok         | input,output,embedding,--epochs                                                   | f1,precision,recall,train_time,eval_time            | tableA_id,tableB_id,label,prob_class1 |
| [magellan](methods/magellan/README.md)           | ok, no GPU | input,output,--method,--seed                                                      | f1,precision,recall,train_time,eval_time            | tableA_id,tableB_id                   |
| [zeroer](methods/zeroer/README.md)               | ok, no GPU | input,output,--transitivity,--full                                                | f1,precision,recall,train_time,eval_time            | tableA_id,tableB_id                   |
