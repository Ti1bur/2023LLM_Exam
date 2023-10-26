#!/bin/bash

target_folder="./tmp/split_token"

for i in $(seq 0 7000000 1000)
do
    folder_name=$(printf "%05d" $i)
    folder_path="$target_folder/$folder_name"
    
    if [ ! -d "$folder_path" ]
    then
        mkdir -p "$folder_path"
    fi
    
    for j in $(seq 0 999)
    do
        file_name="$i.${j}.pkl"
        if [ -e "$file_name" ]
        then
            mv "$file_name" "$folder_path"
        fi
    done
done
