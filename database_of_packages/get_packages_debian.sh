#!/bin/bash

# Get first 2 columns of the output which is delimitates by ' - '
apt-cache search . | awk -F ' - ' '{print $1, "\"" $2 "\""}' > available_packages_debian.txt

while read line; do
    echo $line
    # Get first column of line
    pkg=$(echo $line | awk '{print $1}')
    files=$(apt-file list $pkg | awk -F ': ' '{print $2}' | paste -sd "," -)
    echo $line    $files >> list_of_files_debian.txt
done < available_packages_debian.txt

