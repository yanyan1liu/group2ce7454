#!/bin/bash

# download images
kaggle competitions download -c state-farm-distracted-driver-detection

# download location information
kaggle datasets download -d liuy0156/location-information

# unzip images
unzip imgs.zip
rm imgs.zip
rm driver_imgs_list.csv.zip
rm sample_submission.csv.zip

# unzip location information
unzip location-information.zip
rm location-information.zip
chmod 775 labels.txt
chmod 775 xml.zip 
unzip xml.zip
rm xml.zip

# create training folder
base=group2
dest=$base/images
mkdir -p $dest
mv labels.txt $base

for label in c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
do
    src=train/$label
    echo "$src"

    files=$src/*.jpg
    for imgF in $files
    do      
        xmlF="${imgF//$src/xmls}"
        xmlF="${xmlF//.jpg/.xml}"
           
        if [ -f "$xmlF" ]; then
            mv $imgF $dest
            mv $xmlF $dest
        else
            echo "$imgF cannot find $xmlF"
        fi
    done
done

rm -rf xmls
rm -rf train
rm -rf test

trainD=$dest/0
testD=$dest/1
mkdir -p $trainD
mkdir -p $testD

count=0
files=$dest/*.jpg
for imgF in $files
do  
    (( count++ ))
    xmlF="${imgF//.jpg/.xml}"
    if [ $count -le 20000 ];then
        mv $imgF $trainD
        mv $xmlF $trainD
    else
        mv $imgF $testD
        mv $xmlF $testD
    fi
done