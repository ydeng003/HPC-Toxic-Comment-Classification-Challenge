#!/bin/bash -l
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4

enable_lmod
module load spark
module load python/3.6
module load numpy
start_spark_cluster
spark-submit tfidf.py train.csv test.csv tfidf/train tfidf/test 3000