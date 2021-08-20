Survival regression with accelerated failure time model in XGBoost: Computer Code
=================================================================================
This repository contains the data and code necessary to reproduce the experiments described in
the article *Survival regression with accelerated failure time model in XGBoost*. Each directory
(e.g. `section_3.1`) corresponds to a section in the article.

## Setting up environment
Python 3.8+ and R 3.5.0+ are required to run the provided code.

### Setting up the Python environment

1. Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
2. Set up a new Conda environment by running `conda env create -f environment.yml`. It will
   download and install all necessary Python packages in the Codna environment `xgboost_aft`.
3. Install `mmit` from the source, as it is not yet available from PyPI or Conda:
```
conda activate xgboost_aft
git clone https://github.com/aldro61/mmit.git
cd mmit
python setup.py install
cd ..
```

### Setting up the R environment
1. Ensure that you have installed R 3.5.0 or later.
2. Install the necessary R packages:
```
install.packages(c('survival', 'penaltyLearning', 'rjson', 'future', 'future.apply', 'directlabels'))
```

## Running the experiments from Section 3.1
```
conda activate xgboost_aft
cd section_3.1
./run_xgb_aft.sh
./run_penalty_learning.sh
./run_survreg.sh
./run_mmit.sh
python make_plot.py
cd ..
```
The synthetic datasets have already been generated for you, but if you are
curious about how it was generated, check out `generate_interval_censored_data.py`.

## Running the experiment from Section 3.2
```
cd section_3.2
./run_right_censored_experiment.sh
python make_plot.py
cd ..
```

## Running the experiment from Section 3.4
```
cd section_3.4
./gpu_experiment.sh &> log.txt
cat log.txt
```

## Running the experiment from Section 3.3
The code in Section 3.3 is an adapted version of Section 3.1. It takes great
care to run it on the cloud. Here, we can only give a sketch of what it takes to get it running on
Amazon Web Services:

* Launch a new EC2 instance. Choose Ubuntu 18.04 as the OS type. This will be the manager.
* SSH into the manager instance. Install Miniconda, and then install Python packages `boto3`,
  `asyncssh`, and `awscli`.
* Now launch a new EC2 instance. This will serve as a template for workers.
* SSH into the worker instance. Install Miniconda and Docker. Install Python package `awscli`.
* Stop the worker instance. Once it stops running, make an AMI image.
* Create a new Docker image repository on Elastic Container Registry (ECR).
* Using the `aft/Dockerfile`, build a Docker image and then put it to the ECR repository.
* Create a new S3 bucket, to store the logs from the experiment.
* Create a new IAM instance profile called `AFTWorkerRole`. To this profile, attach permissions to
  do the following: Pull from ECR, Write to S3
* Create a new IAM instance profile called `AFTManagerRole`. To this profile, attach permissions to
  do the following: Launch and Terminate EC2 instances.
* Attach `AFTManagerRole` to the manager instance.
* SSH into the manager instance again. Copy the code from the directory `section_3.3` into the
  manager instance.
* Edit `launcher.py` to use the correct SSH key and ECR repository, S3 bucket, and most importantly
  the ID of the AMI image of the worker.

Please e-mail chohyu01@cs.washington.edu if you run into any troubles.
