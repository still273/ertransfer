#!/bin/bash

# exit on ctrl+c
trap "exit" INT

# "d1_fodors_zagats" "d2_abt_buy" "d3_amazon_google" "d4_dblp_acm" "d5_imdb_tmdb" "d6_imdb_tvdb" "d7_tmdb_tvdb" "d8_amazon_walmart" "d9_dblp_scholar"
# run ditto on all datasets

# sbatch --job-name "ditto-d2_abt_buy" ./ditto.sh "../../datasets/d2_abt_buy/kj_split/" "../../output/ditto/d2_abt_buy/full/" -t "../../datasets/d3_amazon_google/kj_split/" "../../datasets/d8_amazon_walmart/kj_split/" -e 40 -if -tf
 #     -t "../../datasets/d10_imdb_dbpedia/kj_split"

#sbatch --job-name "ditto_d5" ./ditto.sh "../../datasets/d5_imdb_tmdb/kj_split/" "../../output/ditto/d5_imdb_tmdb/full/" -e 40 -if -tf -pt \
#  -t "../../datasets/d10_imdb_dbpedia/kj_split"

for ds in  "d3_amazon_google" "d6_imdb_tvdb" "d7_tmdb_tvdb" "d8_amazon_walmart" "d10_imdb_dbpedia"; do
    sbatch --job-name "ditto-$ds" ./ditto.sh "../../datasets/$ds/kj_split/" "../../output/ditto/$ds/split/" -e 40
done
