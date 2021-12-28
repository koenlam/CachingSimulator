
echo "[+] Downloading Movielens dataset"
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip

echo "[+] Unzipping Movielens dataset"
unzip ml-1m.zip

echo "[+] Renaming Movielens dataset"
mv ml-1m/ratings.dat MovieLens1M_ratings.dat

echo "[+] Finished downloading the Movielens dataset"