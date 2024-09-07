import argparse
import json
import os
import pickle
import subprocess

from collections import Counter

OUTPUT_MAP_FILE = 'FILE_MAP.pkl'


def accepted_pkg(pkg):
    if 'langpack' in pkg:
        return False
    if pkg.endswith('-doc'):
        return False
    return True

def accepted_file(file):
    file_name = file.split('/')[-1].strip()

    # Exclude if it doesn't start with a letter
    if not file_name[0].isalpha():
        return False
    # Exclude if it starts with a digit
    if file_name[0].isdigit():
        return False
    # Exclude if it contains a space
    if ' ' in file:
        return False
    
    if 'langpack' in file:
        return False
    if file_name == 'config':
        return False

    exclude_extensions = [
        'html',
        'txt',
        'md',
        'rst',
        'gz',
        'zip',
        'tar',
        'bdf',
        'htf',
        'qml',
        'xml',
        'png',
        'jpg',
        'jpeg',
        'gif',
        'svg',
        'ico',
        'pdf',
        'doc',
        'docx',
        'xls',
        'md5',
        'sha1',
        'sha256',
        'sha512',
        'crt',
        'pem',
        'key',
        'cer',
        'conf'
    ]
    for ext in exclude_extensions:
        if file.endswith('.' + ext):
            return False
    return True


def main(input_file, source, reset, json_output):
    files_map = {}
    if not reset:
        with open(OUTPUT_MAP_FILE, 'rb') as fin:
            files_map = pickle.load(fin)

    all_files = []
    with open(input_file, 'rt') as fin:
        for line in fin:
            splits = line.split()
            if len(splits) < 4:
                continue
            if accepted_pkg(splits[0]):
                all_files.extend(splits[1].split(','))

    counter = Counter(all_files)

    with open(input_file, 'rt') as fin:
        for line in fin:
            splits = line.split()
            if len(splits) < 4:
                continue
            pkg = splits[0]
            version = splits[2]
            summary = ' '.join(splits[3:])
            if not accepted_pkg(pkg):
                continue
    
            files = splits[1].split(',')
            for file in files:
                if not accepted_file(file) or counter[file] != 1:
                    continue
                file_name = file.split('/')[-1].strip()
                if file_name not in files_map:
                    files_map[file_name] = {}
                if file not in files_map[file_name]:
                    files_map[file_name][file] = []
                files_map[file_name][file].append({'src': source, 'pkg': pkg, 'version': version, 'summary': summary})

    with open(OUTPUT_MAP_FILE, 'wb') as fout:
        pickle.dump(files_map, fout)

    if json_output:
        with open('tmp.json', 'wt') as fout:
            json.dump(files_map, fout, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import new source for packages')
    parser.add_argument('source', help='Name of source for packages')
    parser.add_argument('--reset', action='store_true', help='Reset the file map', default=False)
    parser.add_argument('--json', action='store_true', help='Output the result as JSON', default=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--script', help='Path to the script that generates the list of files')
    group.add_argument('--file', help='Path to the file that contains the list of files')
    args = parser.parse_args()

    if args.script:
        print("Running the script to get the list of packages and their files")
        subprocess.run(['/bin/bash', args.script])
        main(f'list_of_files_{args.source}.txt', args.source, args.reset, args.json)
    else:
        main(args.file, args.source, args.reset, args.json)