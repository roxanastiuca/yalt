# Job labelling
Requires CSV files with data about cluster jobs: list of files opened, and binaries and the strings contained.


## Setup
```
python -m venv myenv
source myenv/bin/activate
python -m pip install -r requirements.txt
```

## Use
First python script matches job data with data in the database of knowledge, and finds the packages linked dynamically and statically into the program. Second script summarizes the information, tries to cluster the jobs, and creates word clouds from packages name and summaries.
```
python extract_packages.py
python extract_labels.py
```