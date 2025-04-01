# End-to-End Deep Entity Resolution with No Labelled Instances

We propose a novel methodology for end-to-end Deep ER that waives the need for labelled instances. In essence, 
it associates the pair of data sources at hand D1 and D2 with the another pair of data sources of known groundtruth, 
d1 and d2, whose similarity distributions after Filtering are very close. The Verification method is then trained
on the candidate pairs of d1, d2 and applied to the candidate pairs of D1, D2, linking their records without any 
labelled instances. Through a thorough experimental study that involves six datasets from two different domains, 
we demonstrate the high effectiveness of our approach, which outperforms established methods that require no 
labelled instances, while approximating the performance of the state-of-the-art Verification algorithms.

## Repository setup

```bash
git clone https://github.com/still273/ertransfer.git
git submodule update --recursive --init
```

### Download embeddings

```bash
mkdir embedding
wget https://zenodo.org/record/6466387/files/wiki.en.bin -O embedding/wiki.en.bin
```

## Running Instructions:

### Filtering:
Filtering Methods are contained in the folder splitters. Instructions to set up the apptainer can be found in the 
corresponding readme. Running the apptainer executes splitter.py. As arguments, it needs an input and an output directory.
The optional argument `-d` determines if the default settings for the Filtering method should be used.

### Matching:
All Matching algorithms can be found in the folder methods. Instructions to set up the apptainer can be found in the
corresponding readmes. Running the apptainer executes entrypoint.py, which take as arguments an input folder for the
training dataset and an output folder. Further important parameters are:
- `-e`: number of epochs
- `-if`: how much of the labelled dataset should be used for training
- `-tf`: how much of the unlabelled dataset should be used for inference
- `-pt`: should the saved model be used for inference or if no model is saved, save the trained model
- `-le`: save model at last epoch instead of at the best validation F1
- `-t`: list of folders containing datasets for inference

### Clustering:
Clustering algorithms can be found in the folder clustering/Probabilities. Instructions to set up teh apptainer can be found
in the corresponding readme. Running the apptainer executes entrypoint.py, which takes as an argument  an input file,
containing the matching probability of the candidate pairs. This is produced by the matching algorithms.
The output is saved in the same folder as the input. The option `-d` determines whether the default threshold should be
used or the threshold should be finetuned.
