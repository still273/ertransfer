#!/bin/bash

# exit on ctrl+c
trap "exit" INT

# "d1_fodors_zagats" "d2_abt_buy" "d3_amazon_google" "d4_dblp_acm" "d5_imdb_tmdb" "d6_imdb_tvdb" "d7_tmdb_tvdb" "d8_amazon_walmart" "d9_dblp_scholar"
# run gnem on all datasets
#sbatch --job-name "gnem_d5" ./gnem.sh "../../datasets/d5_imdb_tmdb/kj_split/" "../../output/gnem/d5_imdb_tmdb/full/" -e 40 -if -tf -pt
sbatch --job-name "gnem_d2" ./gnem.sh "../../datasets/d2_abt_buy/kj_split/" "../../output/gnem/d2_abt_buy/full/" -e 40 -if -tf -pt \
  -t "../../datasets/d3_amazon_google/kj_split/" "../../datasets/d8_amazon_walmart/kj_split/"
#sbatch --job-name "gnem_d5" ./gnem.sh "../../datasets/d5_imdb_tmdb/kj_split/" "../../output/gnem/d5_imdb_tmdb/full/" -e 40 -if -tf -pt

#for ds in  "d3_amazon_google" "d6_imdb_tvdb" "d7_tmdb_tvdb" "d8_amazon_walmart" "d10_imdb_dbpedia"; do
#    sbatch --job-name "gnem-$ds" ./gnem.sh "../../datasets/$ds/kj_split/" "../../output/gnem/$ds/split/" -e 40
#done
