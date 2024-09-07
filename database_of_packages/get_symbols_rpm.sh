#!/bin/bash

# For each line in available_packages_rpm.txt
while read line; do
    echo $line
    mkdir rpm-extraction
    cd rpm-extraction
    # Download the rpm file
    sudo dnf download $line.x86_64
    # Extract the rpm file
    FILES=$(rpm2cpio $line*.rpm | cpio -idmv 2>&1)
    TR_FILES=$(echo $FILES | awk '{ for (i=1; i<=NF-2; i++) printf "%s\n", $i }')
    for file in $TR_FILES; do
        # Check file exists
        if [ ! -f $file ]; then
            continue
        fi
        # Skip if mime-type doesn't include 'sharedlib', 'script' or 'executable'
        if ! file --mime-type $file | grep -q 'sharedlib\|script\|executable'; then
            continue
        fi
        # Get the symbols from the file
        out_strings=$(strings -n 10 $file | sed 's/"/""/g; s/,/\\,/g; s/'\''/\\'\''/g' | paste -sd,)
        # Get the symbols from the file
        out_nm=$(nm --format=posix --demangle $file | awk '{print $1}' | sed 's/"/""/g; s/,/\\,/g; s/'\''/\\'\''/g' | paste -sd,)
        # Print to CSV
        echo "rpm,$line,$file,\"$out_nm\",\"$out_strings\"" >> ../SYMBOLS.csv
    done
    cd ..
    rm -rf rpm-extraction
done < available_packages_rpm.txt
