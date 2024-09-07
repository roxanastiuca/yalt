import argparse
import pickle
import json

OUTPUT_MAP_FILE = 'SUMMARIES_MAP.pkl'


def extract_summaries(input_file, source, reset, json_output):
    summaries_map = {}
    if not reset:
        with open(OUTPUT_MAP_FILE, 'rb') as fin:
            summaries_map = pickle.load(fin)

    with open(input_file, 'rt') as fin:
        for line in fin:
            splits = line.split()
            if len(splits) < 4:
                continue
            pkg = splits[0]
            version = splits[2]
            summary = ' '.join(splits[3:])
            summaries_map[pkg]= {'src': source, 'version': version, 'summary': summary}

    with open(OUTPUT_MAP_FILE, 'wb') as fout:
        pickle.dump(summaries_map, fout)

    if json_output:
        with open('tmp_summary.json', 'wt') as fout:
            json.dump(summaries_map, fout, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract summaries from a file')
    parser.add_argument('input_file', help='Input file')
    parser.add_argument('--source', help='Name of source for packages', required=True)
    parser.add_argument('--reset', action='store_true', help='Reset summaries')
    parser.add_argument('--json', action='store_true', help='Output summaries in JSON format')
    args = parser.parse_args()

    extract_summaries(args.input_file, args.source, args.reset, args.json)