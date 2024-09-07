#!/bin/bash

while read line; do
    echo $line
    pkg=$(echo $line | awk '{print $1}')
    data=$(dnf repoquery --qf "%{version} \"%{summary}\"" $pkg | tail -n 1)
    echo $line $data >> rpm.txt
done < list_of_files_rpm.txt