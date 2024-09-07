#!/bin/bash

total=$(wc -l available_packages_debian.txt | awk '{print $1}')
count=0
# For each line in available_packages_debian.txt
while read line; do
    pkg=$(echo $line | awk '{print $1}')
    count=$((count+1))
    # # Skip if count is less than 4845
    # if [ $count -lt 22241 ]; then
    #     continue
    # fi
    echo "$count/$total: $pkg"
    mkdir debian-extraction
    cd debian-extraction
    # Download the deb file
    sudo apt download $pkg
    # Extract description and version
    description=$(dpkg -I $pkg*.deb | grep -m 1 Description | sed 's/Description: //' | cut -c2-)
    version=$(dpkg -I $pkg*.deb | grep -m 1 Version | sed 's/Version: //' | cut -c2-)
    # Extract the deb file
    FILES=$(dpkg -X $pkg*.deb . 2>/dev/null)
    files_out=""
    for file in $FILES; do
        # Check file exists
        if [ ! -f $file ]; then
            continue
        fi
        # Skip if mime-type doesn't include 'sharedlib', 'script' or 'executable'
        if ! file --mime-type $file | awk '{print $2}' | grep -q 'sharedlib\|script\|executable'; then
            continue
        fi
        # Add file to list
        files_out="$files_out,$(echo $file | cut -c2-)"
        # Get the symbols from the file
        out_strings=$(strings -n 10 $file | sed 's/"/""/g; s/,/\\,/g; s/'\''/\\'\''/g' | paste -sd,)
        # Get the symbols from the file
        out_nm=$(nm --format=posix --demangle $file | awk '{print $1}' | sed 's/"/""/g; s/,/\\,/g; s/'\''/\\'\''/g' | paste -sd,)
        # Print to CSV
        echo "debian,$pkg,$file,\"$out_nm\",\"$out_strings\"" >> ../SYMBOLS.csv
    done
    # If $files_out length more than 1, remove the first comma
    if [ ${#files_out} -gt 1 ]; then
        echo $pkg $(echo $files_out | cut -c2-) $version \"$description\" >> ../DEBIAN.txt
    fi
    cd ..
    rm -rf debian-extraction
done < available_packages_debian.txt
