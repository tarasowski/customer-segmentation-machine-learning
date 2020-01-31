# Customer Segmentation (Unsupervised Learning)

The goal of this project is to apply unsupervised machine learning techniques to identify
segments that form the customer base of a mail-order company in Germany. These
segments can be used to create direct marketing campaigns with the highest
return on investment.

## Pre-requisites
* Python v3.7
* Scikit-learn v0.21.3
* Joblib v0.14.0
* Numpy v1.17.2
* Pandas v0.25.1


## Getting Started
Every component of the ML pipeline is properly decomposed. There are three main
parts of the project: preprocess, train, predict. Each file / part can run
independently. Unfortunately due to project constraints (license issues) the data provided by
Arvato cannot be uploaded to this repository. It means there is no way to run
the pipeline just by cloning the repository. 

The first step is to preprocess the data, run the following command. 

* Basic usage: `./run preprocess` 

The next step is to train the model, run the following commands. 

* Basic usage: `./run train` 
* Options:
  * Choose number of PCA components: `--pca_components n`
  * Choose number of clusters: `--n_clusters n`

The last step is to make predictions based on the trained model, run the following
command:

* Basic usage: `./run predict`

At every step, some required arguments need to be supplied. These arguments are predefined in
the `./run` file. 

## Support
Patches are encouraged and may be submitted by forking this project and
submitting a pull request through GitHub.

## Credits
The project was developed during the ML program of
[Udacity.com](https://www.udacity.com/)

## Licence
Released under the [MIT License](./License.md)

