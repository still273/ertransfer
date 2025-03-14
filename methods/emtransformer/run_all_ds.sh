#!/bin/bash

# exit on ctrl+c
trap "exit" INT

# "d1_fodors_zagats"  "d2_abt_buy" "d3_amazon_google" "d4_dblp_acm" "d5_imdb_tmdb" "d6_imdb_tvdb" "d7_tmdb_tvdb" "d8_amazon_walmart" "d9_dblp_scholar" "d10_imdb_dbpedia"
# run emtransformer on all datasets
#sbatch --job-name "emt_d5" ./emtransformer.sh "../../datasets/d5_imdb_tmdb/kj_split/" "../../output/emtransformer/d5_imdb_tmdb/full/" -e 40 -if -tf -pt \
#  -t "../../datasets/d10_imdb_dbpedia/kj_split"

for ds in  "d3_amazon_google" "d6_imdb_tvdb" "d7_tmdb_tvdb" "d8_amazon_walmart" "d10_imdb_dbpedia"; do
    sbatch --job-name "emtransformer-$ds" ./emtransformer.sh "../../datasets/$ds/kj_split/" "../../output/emtransformer/$ds/split/" -e 40
done
