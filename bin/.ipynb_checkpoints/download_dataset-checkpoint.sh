#!/usr/bin/env bash

mkdir -p data
cd data
if [ ! -f "CrisisMMD_v2.0.tar.gz" ]; then 
  wget https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz
fi
if [ ! -f "crisismmd_datasplit_all.zip" ]; then 
  wget https://crisisnlp.qcri.org/data/crisismmd/crisismmd_datasplit_all.zip
fi
if [ ! -f "crisismmd_datasplit_agreed_label.zip" ]; then 
  wget https://crisisnlp.qcri.org/data/crisismmd/crisismmd_datasplit_agreed_label.zip
fi
rm -rf __MACOSX
rm -rf crisismmd_datasplit_agreed_label
unzip crisismmd_datasplit_agreed_label.zip
rm -rf crisismmd_datasplit_all
unzip crisismmd_datasplit_all.zip
tar xzvf CrisisMMD_v2.0.tar.gz
rm -rf CrisisMMD_images
mkdir -p CrisisMMD_images
cp CrisisMMD_v2.0/data_image/*/*/* CrisisMMD_images/
cd ..
