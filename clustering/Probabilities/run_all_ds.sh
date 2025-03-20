#!/bin/bash

# exit on ctrl+c
trap "exit" INT

# "d1_fodors_zagats" "d2_abt_buy" "d3_amazon_google" "d4_dblp_acm" "d5_imdb_tmdb" "d6_imdb_tvdb" "d7_tmdb_tvdb" "d8_amazon_walmart" "d9_dblp_scholar"
# run ditto on all datasets

# sbatch --job-name "ditto-d2_abt_buy" ./ditto.sh "../../datasets/d2_abt_buy/kj_split/" "../../output/ditto/d2_abt_buy/full/" -t "../../datasets/d3_amazon_google/kj_split/" "../../datasets/d8_amazon_walmart/kj_split/" -e 40 -if -tf
 #     -t "../../datasets/d10_imdb_dbpedia/kj_split"

#sbatch --job-name "ditto_d5" ./ditto.sh "../../datasets/d5_imdb_tmdb/kj_split/" "../../output/ditto/d5_imdb_tmdb/full/" -e 40 -if -tf -pt \
#  -t "../../datasets/d10_imdb_dbpedia/kj_split"

#for ds in  "d3_amazon_google" "d8_amazon_walmart"; do
#    sbatch --job-name "cluster_f-$ds" ./clustering.sh "../../output/emtransformer/d2_abt_buy/full_last_all/predictions_$ds.csv"
#done
#
#for ds in "d6_imdb_tvdb" "d7_tmdb_tvdb" "d10_imdb_dbpedia"; do
#    sbatch --job-name "cluster_f-$ds" ./clustering.sh "../../output/emtransformer/d5_imdb_tmdb/full_last_all/predictions_$ds.csv"
#done


for method in 'ditto' 'emtransformer' 'gnem'; do
  for ds in "d2_abt_buy" "d3_amazon_google" "d6_imdb_tvdb" "d7_tmdb_tvdb" "d8_amazon_walmart"; do
    sbatch --job-name "cluster_$method" ./clustering.sh "../../output/$method/dkj_split/d5_imdb_tmdb/full/predictions_$ds.csv" -d
  done
done
for ds in "d2_abt_buy" "d5_imdb_tmdb"  "d3_amazon_google" "d8_amazon_walmart" "d6_imdb_tvdb" "d7_tmdb_tvdb"; do
    sbatch --job-name "cluster_s-$ds" ./clustering.sh "../../output/zeroer/$ds/dkj_split/predictions.csv" -d
done
for method in 'ditto' 'emtransformer' 'gnem'; do
  for ds in "d2_abt_buy" "d3_amazon_google" "d5_imdb_tmdb" "d6_imdb_tvdb" "d7_tmdb_tvdb" "d8_amazon_walmart"; do
    sbatch --job-name "cluster_$method" ./clustering.sh "../../output/$method/dkj_split/$ds/split/predictions_$ds.csv" -d
  done
done

