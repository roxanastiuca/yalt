#!/bin/bash

dnf list available | tail -n +3 | awk '{print $1}' > available_packages_rpm.txt
while read line; do
    echo $line
    files=$(dnf repoquery --list $line | paste -sd "," -)
    echo $line    $files >> list_of_files_rpm.txt
done < available_packages_rpm.txt
rm available_packages_rpm.txt