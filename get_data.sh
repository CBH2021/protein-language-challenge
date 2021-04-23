#!/bin/bash
mkdir data
wget "https://www.dropbox.com/s/wbf1ez145i4dtwn/TS115_ESM1b.npz?dl=0" -O TS115_ESM1b.npz
mv TS115_ESM1b.npz data/TS115_ESM1b.npz

wget "https://www.dropbox.com/s/dpar81d79ms1sog/Train_ESM1b.npz?dl=0" -O Train_ESM1b.npz
mv Train_ESM1b.npz data/Train_ESM1b.npz
