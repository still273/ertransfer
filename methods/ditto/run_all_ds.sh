#!/bin/bash

# exit on ctrl+c
trap "exit" INT

# "d1_fodors_zagats" "d2_abt_buy" "d3_amazon_google" "d4_dblp_acm" "d5_imdb_tmdb" "d6_imdb_tvdb" "d7_tmdb_tvdb" "d8_amazon_walmart" "d9_dblp_scholar"
# run ditto on all datasets

sbatch --job-name "ditto-d2_abt_buy" ./ditto.sh "../../datasets/d2_abt_buy/kj_split/" "../../output/ditto/d2_abt_buy/" -t "../../datasets/d3_amazon_google/kj_split/" "../../datasets/d8_amazon_walmart/kj_split/" -e 20

