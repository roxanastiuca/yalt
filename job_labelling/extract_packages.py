import argparse
import heapq
import json
import numpy as np
import pandas as pd
import os
import pickle
from collections import Counter
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
import re
import string


FILE_MAP_LOCATION = '../database_of_packages/FILE_MAP.pkl'
SUMMARIES_MAP_LOCATION = '../database_of_packages/SUMMARIES_MAP.pkl'
SYMBOLS_LOCATION = '../database_of_packages/SYMBOLS.csv'

STOP_WORDS = set(stopwords.words('english'))


file_mapping = {}
summary_mapping = {}
def init_mapping():
    global file_mapping
    global summary_mapping
    with open(FILE_MAP_LOCATION, 'rb') as fin:
        file_mapping = pickle.load(fin)
    with open(SUMMARIES_MAP_LOCATION, 'rb') as fin:
        summary_mapping = pickle.load(fin)


def remove_suffix(p):
    p = p.lower()
    for suffix in ['.x86_64', '.i686', '.noarch']:
        if p.endswith(suffix):
            p = p[:-len(suffix)]
    return p


def tokenize_package(p):
    p = remove_suffix(p)
    s = re.split(r'[{}]'.format(string.punctuation + r"\'\"\\\\" + string.digits), p)
    s = [w for w in s if w not in ['', 'lib']]
    s.extend([w.removeprefix('lib') for w in s if w.startswith('lib')])
    return s


def tokenize_summary(s):
    s = s.lower()
    s = re.sub(r'[{}]'.format(string.punctuation + r"\'\"\\\\"), '', s)

    return [w for w in s.split() if w not in STOP_WORDS]


def find_packages_for_file(file_path, count):
    pkgs = []
    pkgs_short = []
    file_name = file_path.split('/')[-1]

    if '.so.' in file_name:
        # Take only until the trailing version numbers
        file_name = file_name.split('.so')[0] + '.so'

    if file_name in file_mapping.keys():
        curr_data = {
            'file': file_path,
            'count': count,
        }

        possibilites = file_mapping[file_name]
        all_pkgs = []
        all_summaries = []

        if file_path in possibilites:
            curr_data['pkg'] = possibilites[file_path]
            all_pkgs.extend([p["pkg"] for p in possibilites[file_path]])
            all_summaries.extend([p["summary"] for p in possibilites[file_path]])
        else:
            curr_data['pkg'] = None
            curr_data['possibilities'] = possibilites
            for p in possibilites.values():
                all_pkgs.extend([pp["pkg"] for pp in p])
                all_summaries.extend([pp["summary"] for pp in p])

        if len(all_pkgs) > 1:
            # Get common strings between all pkgs
            common_pkgs = set.intersection(*[set(
                tokenize_package(p)
            ) for p in all_pkgs])
        else:
            common_pkgs = [remove_suffix(p) for p in all_pkgs]
            common_summaries = all_summaries

        # Get common strings between all summaries
        common_summaries = set.intersection(*[set(
            tokenize_summary(s)
        ) for s in all_summaries])

        curr_data['pkg_common'] = list(common_pkgs)
        curr_data['summary_common'] = list(common_summaries)

        pkgs.append(curr_data)
        # Delete the 'possibilities' key
        data_short = curr_data.copy()
        if 'possibilities' in data_short:
            del data_short['possibilities']
        pkgs_short.append(data_short)


    return pkgs, pkgs_short


###### PROCESS AND FILTER STRINGS ######
def process_strings(strings):
    strings = set(strings.split(','))
    strings = {
        s for s in strings
        if len(s) > 2
            and not s.startswith('.')
            and not s.startswith('_')
            and not s.startswith('GLIBC')
            and not s.startswith('CXX')
    }
    return strings

###### FIND PACKAGES FOR BINARIES ######
binary_funcs = ['jaccard', 'proportion']

def find_packages_for_binary_jaccard(row):
    bin_pkgs = []

    # If row['comm_path'] is not a valid string (str, and len>2), return empty list
    if not isinstance(row['comm_path'], str) or len(row['comm_path']) < 2:
        return bin_pkgs

    print(f'Finding packages for {row['comm_path']}')
    comm_path = '.' + row['comm_path'] # Add a dot to match the format in the database

    strings_truth = set(row['strings'].split(','))
    top_3_similar = []

    chunksize = 10 ** 6
    # for chunk in tqdm(pd.read_csv(SYMBOLS_LOCATION, chunksize=chunksize), desc='Processing chunks'):
    for chunk in tqdm(pd.read_csv(SYMBOLS_LOCATION, chunksize=chunksize), desc='Processing chunks'):
        for index, r in tqdm(chunk.iterrows(), total=chunk.shape[0], desc='Processing rows', leave=False):
            # Compute Jaccard similarity between strings_truth and r['strings']
            if not isinstance(r['strings'], str) or len(r['strings']) < 2:
                continue
            strings = set(r['strings'].split(','))
            if len(strings) < 2:
                continue
            jaccard = len(strings_truth.intersection(strings)) / len(strings_truth.union(strings))
            if len(top_3_similar) < 3:
                heapq.heappush(top_3_similar, (jaccard, [r['source'], r['package'], r['file']]))
            else:
                heapq.heappushpop(top_3_similar, (jaccard, [r['source'], r['package'], r['file']]))

    for jaccard, r in top_3_similar:
        bin_pkgs.append({'comm_path': row['comm_path'], 'source': r[0], 'pkg': r[1], 'file': r[2], 'jaccard': jaccard})


    return bin_pkgs


# Proportion of elements from strings in a package that are present in the binary
def find_packages_for_binary_proportion(row):
    bin_pkgs = []
    bin_pkgs_short = []

    # If row['comm_path'] is not a valid string (str, and len>2), return empty list
    if not isinstance(row['comm_path'], str) or len(row['comm_path']) < 2:
        return bin_pkgs, bin_pkgs_short

    print(f'Finding packages for {row['comm_path']}')
    comm_path = '.' + row['comm_path'] # Add a dot to match the format in the database

    strings_truth = process_strings(row['strings'])

    chunksize = 10 ** 6
    for chunk in tqdm(pd.read_csv(SYMBOLS_LOCATION, chunksize=chunksize), desc='Processing chunks'):
        for index, r in tqdm(chunk.iterrows(), total=chunk.shape[0], desc='Processing rows', leave=False):
            # Compute Jaccard similarity between strings_truth and r['strings']
            if not isinstance(r['strings'], str) or len(r['strings']) < 2:
                continue
            strings = process_strings(r['strings'])
            if len(strings) < 2:
                continue
            intersection = strings_truth.intersection(strings)
            proportion = len(intersection) / min(len(strings), len(strings_truth))
            if len(intersection) < 25 and proportion < 0.95:
                continue

            if proportion > 0:
                summary = summary_mapping[r['package']]['summary'] if r['package'] in summary_mapping else ''
                bin_pkgs.append(
                    {
                        'comm_path': row['comm_path'],
                        'source': r['source'],
                        'pkg': r['package'],
                        'summary': summary,
                        'summary_common': tokenize_summary(summary),
                        'pkg_common': [remove_suffix(r['package'])],
                        'file': r['file'],
                        'proportion': proportion,
                        'intersection': list(intersection)
                    })
                bin_pkgs_short.append(
                    {
                        'comm_path': row['comm_path'],
                        'source': r['source'],
                        'pkg': r['package'],
                        'summary': summary,
                        'summary_common': tokenize_summary(summary),
                        'pkg_common': [remove_suffix(r['package'])],
                        'file': r['file'],
                        'proportion': proportion
                    })


    return bin_pkgs, bin_pkgs_short

###### MAIN FUNCTION ######
def main(events, binaries, jobid, binary_func):
    # Extract only events with jobid
    events = events[events['jobid'] == jobid]
    events = events[events['ret'] >= 0]
    binaries = binaries[binaries['jobid'] == jobid]

    print("Processing:")
    print(binaries[['jobid', 'label', 'comm', 'comm_path']])

    # For each row in events, call find_packages
    files = []
    for index, row in events.iterrows():
        files.append((row['file_path'], row['comm']))
    files_counter = Counter(files)
    pkgs = []
    pkgs_short = []
    for x, count in files_counter.items():
        file, comm = x
        if comm in ['srun']:
            continue
        x, y = find_packages_for_file(file, count)
        pkgs.extend(x)
        pkgs_short.extend(y)

    # For each row in binaries, call find_packages
    find_packages_for_binary = globals()[f'find_packages_for_binary_{binary_func}']

    bin_pkgs = []
    bin_pkgs_short = []
    bin_processed = set()
    for index, row in binaries.iterrows():
        if row['comm_path'] in bin_processed or row['comm'] in ['srun'] or row['comm_path'] in ['/usr/bin/basename', '/usr/bin/sh']:
            continue
        x, y = find_packages_for_binary(row)
        bin_pkgs.extend(x)
        bin_pkgs_short.extend(y)
        bin_processed.add(row['comm_path'])

    data = {
        'jobid': jobid,
        'label': events['label'].iloc[0],
        'binaries': list(set(binaries['comm_path'])),
        'packages_in_files': pkgs,
        'packages_in_binaries': bin_pkgs
    }

    data_short = {
        'jobid': jobid,
        'label': events['label'].iloc[0],
        'binaries': list(set(binaries['comm_path'])),
        'packages_in_files': pkgs_short,
        'packages_in_binaries': bin_pkgs_short
    }

    # Make directory 'output_{binary_func}' if it doesn't exist
    if not os.path.exists(f'output_{binary_func}'):
        os.makedirs(f'output_{binary_func}')

    with open(f'output_{binary_func}/job_{jobid}.json', 'w') as fout:
        json.dump(data, fout, indent=4)
    
    with open(f'output_{binary_func}/job_{jobid}_short.json', 'w') as fout:
        json.dump(data_short, fout, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract packages')
    parser.add_argument('--events', type=str, help='Events file', default='events.csv')
    parser.add_argument('--binaries', type=str, help='Binaries file', default='binaries.csv')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--jobid', type=int, help='Job ID')
    group.add_argument('--all', action='store_true', help='All jobs')
    parser.add_argument('--binary_func', type=str, help='Binary function',
                        choices=binary_funcs, default=binary_funcs[0])
    args = parser.parse_args()


    init_mapping()

    events = pd.read_csv(args.events)
    binaries = pd.read_csv(args.binaries)

    if args.all:
        for jobid in binaries['jobid'].unique():
            main(events, binaries, int(jobid), args.binary_func)
    else:
        main(events, binaries, args.jobid, args.binary_func)