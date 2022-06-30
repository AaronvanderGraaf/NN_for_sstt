# NN for same-sign top-quark production

## Steps
This project consists of three steps. Firstly convert the root files to .h5 format. Secondly, merge all of them in one .h5 and finally training with notebook.

## 1- Converting the ntupels to .h5
In an empty directory, follow the instructions given in a) for the first time, or b) for the other times to initilise.
```
a)
mkdir FancyDirectory
cd FancyDirectory
mkdir build source
cp -r <pathtothepackage>/HDF5Utils_hacked/HDF5Utils source/
cp <pathtothepackage>/HDF5Utils_hacked/setup.sh source/
cd source
setupATLAS
asetup 21.2.156,AthAnalysis,here
cd ../build
cmake ../source
cmake --build ./
source */setup.sh
```
```
b)
cd FancyDirectory/source/
source setup.sh
```
There are several example scripts on how to run the tool, with selections etc. Please check the folder `NN_for_sstt\HDF5Utils_hacked` for the scripts. These scripts should work out of the box with exception that the path to the building folder has to be changed!

## 2- Preprocessing
This step will merge the files from first step and choose only the given branches. An example how the branch list is given is `preprocess/feature_lists/feature_list_sstt.txt` and the file list format here `preprocess/filelists/filelist_sstt.txt` (class number | name | path). The script can be run as follows:
```
source cvmfs-setup.sh
python preprocess.py -i ./filelists/feature_list_sstt.txt --outdir ./output -f ./feature_lists/filelist_sstt.txt
```

## 3- Training
Finally the training will use the file from step 2. In the folder `NN` there are several notebooks which can be used for training a NN depending on the task. There are notebooks with distance correlation, or without, or for training Signal vs Signal etc. A correct conda envoirment file will be added later.
