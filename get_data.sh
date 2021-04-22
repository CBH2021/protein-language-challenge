#!/bin/bash
mkdir data
wget "https://www.dropbox.com/s/xbrnv1mpl4sjg26/CASP12_ESM1b.npz?dl=0" -O CASP12_ESM1b.npz
mv CASP12_ESM1b.npz data/CASP12_ESM1b.npz

wget "https://www.dropbox.com/s/dpar81d79ms1sog/Train_ESM1b.npz?dl=0" -O Train_ESM1b.npz
mv Train_ESM1b.npz data/Train_ESM1b.npz
