#!/bin/bash

sudo apt-get update
sudo apt-get install gfortran
sudo apt-get install libblas-dev liblapack-dev
#sudo apt-get install libcr-dev mpich2
#sudo apt-get install mpich libmpich-dev # libhdf5-mpich-dev
sudo apt-get install libopenmpi-dev openmpi-bin # libhdf5-openmpi-dev
sudo apt-get install default-jre
