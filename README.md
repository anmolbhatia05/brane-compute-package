[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6612509.svg)](https://doi.org/10.5281/zenodo.6612509)
[![CI](https://github.com/web-services-and-cloud-based-systems/brane-compute-package/actions/workflows/main.yml/badge.svg)](https://github.com/web-services-and-cloud-based-systems/brane-compute-package/actions/workflows/main.yml)

# brane_compute

## What does this package do ? 
This package is used to perform the compute parts of the [titanic kaggle contest](https://www.kaggle.com/competitions/titanic). Namely -   
1. Mounting of datasets on the persistent folder (/data)
2. Printing the shape of the dataset
3. Preprocessing (aggreagtions + feature engineering)
4. Modelling (+trianing)
5. Accuracy 


## Steps to build and run this package 
Follow this [link](https://wiki.enablingpersonalizedinterventions.nl/admins/installation/get-binaries.html) to download all the dependcies and run the brane instance. (need to have both cli and instance installed and the instance running) and then follow from #6 below.

Or  

Follow these instructions -  
(Assuming you have docker, buildx plugin for docker, docker compose)
1. Download brane repository -  `git clone https://github.com/epi-project/brane.git && cd brane`
2. do - `chmod +x ./make.sh`
3. Download cli for linux - 
`sudo wget -O /usr/local/bin/brane https://github.com/epi-project/brane/releases/latest/download/brane-linux`
  
4. (alternate for #3) For macOS - 
`sudo wget -O /usr/local/bin/brane https://github.com/epi-project/brane/releases/latest/download/brane-darwin`

5. Start instance - `./make.sh start-instance --precompiled`
   
6. Run the following to build the package - `brane build ./container.yml`

7. Push the package to run it remotely in your instance (do `brane login http://127.0.0.1 --user <username>` first) - `brane push brane_compute`

8. Use branescript (import this package in it by adding `import brane_compute;` on the top) or do `brane repl --remote http://127.0.0.1:50053` (If you running on a K8 cluster, use the cluster address instead)

Or 

You can read the next section to import the package directly from this github repository. 

## How to import/download this package from Github ?
Assuming that you have brane cli and brane instance downloaded and deployed on your local/K8 cluster, then you can run the following command -> `brane import web-services-and-cloud-based-systems/brane-compute-package`

This will import the package on your container registry hosted on your local/K8 cluster. Then you can use this package in your branescripts or on the repl. 


## Brane (external) functions list for this package

| Brane function      | Description | Input   | Output | Result
| :---        |    :----:   |          :----: |:----:| ---:|
|`mount`|Function that mounts the train.csv and test.csv files in this brane package in the /data folder|-|Returns string - "done"/"error"|`train.csv` and `test.csv` is available in /data|
| `data_shape`      | Function that returns the shape of the dataframe after reading the data from a file.       | path of the file   | shape|shape of the data|
| `preprocessing`   | Function that preprocesses the dataset (train, test)        | path, isTrain      |Integer - 0 (success)/error code|Preprocesses dataset for models|
|`modelling`|Function to train the model on the basis mode provided. Mode is the identifier for the machine learning model provided. |path_train, path_test, mode|0/error code|output vector|
| `get_model_accuracy`      | Function to check the model accuracy      | name   | output|accuracy pf the model|

There are a few helper functions that are called internally, from within the brane functions:  
1. get_df
2. name_proc
3. imputting_na_values
4. cat_to_num
5. missingAge
6. family

## Automated builds and testing for this package

We have set automated build for this package using github actions and also run tests (pytest) on it. The triggers are `push` and `pull_request`, but, one can run it manually as well. Please refer to the image below to see how. (Click on run workflow(CI))
![](./workflow.png)

## How to run the tests locally ?
Assuming that you are in the root of the repository and downloaded the depencies (in the requirements.txt file), run `pytest -v`.


 
