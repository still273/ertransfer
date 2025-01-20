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

| Name                                             | Status | GPU Container Status             |
| ------------------------------------------------ | ------ | -------------------------------- |
| [splitter-simple](splitter-simple/README.md)     | ok     | ok                               |
| [splitter](splitter/README.md)                   | ok     | ok, but probably gpu is not used |
| [deepmatcher](methods/deepmatcher/README.md)     | ok     | fails                            |
| [ditto](methods/ditto/README.md)                 |        |                                  |
| [emtransformer](methods/emtransformer/README.md) |        |                                  |
| [gnem](methods/gnem/README.md)                   |        |                                  |
| [hiermatcher](methods/hiermatcher/README.md)     |        |                                  |
| [magellan](methods/magellan/README.md)           |        |                                  |
| [zeroer](methods/zeroer/README.md)               |        |                                  |
