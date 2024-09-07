# Database of knowledge about packages

Scripts to build a knowledge base about packages, the files provided, and the symbols contained in the object files.

First, use the `get_packages_SRC.sh` and `get_symbols_SRC.sh` scripts, then `python import_new_source.py`. Finally, use `python extract_summaries.py`. You should have 3 files afterwards:
- FILE_MAP.pkl: python pickle with mapping from a file name to file paths to packages that provide it;
- SYMBOLS.csv: records of object files, the packages that provide them, and the strings within those binary files;
- SUMMARIES_MAP.pkl: python pickle with mapping from a package name to its summary/description.