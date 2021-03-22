# Modeling Age of Acquisition Norms Using Transformer Networks
Repository for reproducing results from our FLAIRS 2021 paper.

## Datasets
For convenience, we have included the preprocessed versions of the Wordbank and Kuperman norms. The notebook ```ProcessKupermanNorms.ipynb``` also contains our preprocessing code.

If you wish to obtain the original datasets, please see [here](http://wordbank.stanford.edu) for Wordbank and [here](http://crr.ugent.be/archives/806) for Kuperman.

## Reproducing Results

```KupermanNormsRegression``` and ```KupermanNormsRegressionBaseline``` contain all code for reproducing results from the transformer embeddings and baseline psycholinguistic features respectively. Please install the following dependencies in your python environment:

 1. pandas
 2. numpy
 3. matplotlib
 4. scipy
 5. scikit-learn
 6. transformers

See the homepage of their respective repositories for installation instructions. For the transformer experiments, we recommend using a CUDA compatible GPU to speed up inference.

## Citation
Information for citing our work will be released after the conference.