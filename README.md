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

| Name                                             | Entrypoint | GPU Container Status            | Output format                  | Assigned to |
| ------------------------------------------------ | ---------- | ------------------------------- | ------------------------------ | ----------- |
| [splitter-simple](splitter-simple/README.md)     | ok         | ok                              | ok                             |             |
| [splitter](splitter/README.md)                   | ok         | ok                              | ok                             |             |
| [deepmatcher](methods/deepmatcher/README.md)     | ok         | ok                              | predictions missing            | Franziska   |
| [ditto](methods/ditto/README.md)                 | ok         | fails, issue with output dir    |                                | Oleh        |
| [emtransformer](methods/emtransformer/README.md) | ok         | ok                              | scores eq 0, predictions empty | Franziska   |
| [gnem](methods/gnem/README.md)                   | ok         | ok                              | predictions missing            | Franziska   |
| [hiermatcher](methods/hiermatcher/README.md)     | ok         | ok                              | predictions missing            | Franziska   |
| [magellan](methods/magellan/README.md)           | ok         | fails, missing key in dataset   |                                | Franziska   |
| [zeroer](methods/zeroer/README.md)               | ok         | expects different input columns |                                | Franziska   |
