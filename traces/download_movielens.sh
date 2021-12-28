#!/bin/bash

if [ -d "traces" ]; then 
    echo "Please run this script inside the traces directory"
    exit 1

fi

echo "[+] Downloading MovieLens dataset"
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip

echo "[+] Unzipping MovieLens dataset"
unzip ml-1m.zip

echo "[+] Moving MovieLens dataset"
mv ml-1m/ratings.dat MovieLens1M_ratings.dat

echo "[-] Removing the leftover files"
rm -r ml-1m
rm ml-1m.zip

echo "[+] Finished downloading the MovieLens dataset"