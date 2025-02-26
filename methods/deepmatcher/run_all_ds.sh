#!/bin/bash

# exit on ctrl+c
trap "exit" INT


# run ditto on all datasets
for ds in "d3_amazon_google" "d5_imdb_tmdb" "d6_imdb_tvdb" "d7_tmdb_tvdb" ; do
    sbatch --job-name "deepmatcher-$ds" ./deepmatcher.sh "../../datasets/$ds/db_split/" "../../output/deepmatcher/$ds/" "../../embedding/" -e 40
done
