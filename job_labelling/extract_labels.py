import argparse
import json
import os
import re
import string
from collections import Counter
from nltk.corpus import stopwords
from rake_nltk import Rake
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

rake = Rake()
kw_model = KeyBERT()
tfidf_vectorizer = TfidfVectorizer()

stop_words = set(stopwords.words('english'))


def process_text(text):
    text = text.replace('libraries', '')
    text = text.replace('library', '')
    text = text.replace('lib', '')
    # Remove special characters, stop words or single characters
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    text = ' '.join([w for w in text.split()
                     if w not in stop_words and len(w) > 1
                     and w not in ['lib', 'libs', 'library', 'libraries', 'shared']
                     ])
    return text


def find_common_pkg_name(entry):
    common = entry['pkg_common']
    for pkg in common:
        if pkg.startswith('lib') and pkg.replace('lib', '') in common:
            common.remove(pkg.replace('lib', ''))
    # Make one package out of tokens in common, in the order they appear in entry['possibilities'][0]['pkg']
    if 'possibilities' in entry and len(entry['possibilities']) > 0:
        pkg = list(entry['possibilities'].values())[0][0]['pkg']
        common = sorted(common, key=lambda x : pkg.index(x))
        return '-'.join(common)
    else:
        return common[0] if common else None
    
def find_common_summary(entry):
    common = entry["summary_common"]

    if 'possibilities' in entry and len(entry['possibilities']) > 0:
        summary = list(entry['possibilities'].values())[0][0]['summary'].lower()
        summary = re.sub(r'[{}]'.format(string.punctuation + r"\'\"\\\\"), '', summary)
        common = sorted(common, key=lambda x : summary.index(x))
        return ' '.join(common)
    else:
        return ' '.join(common) if common else None



def extract_pkgs_and_text_from_job(data):
    pkgs_matched = []
    pkgs = Counter()
    text = []

    for entry in data['packages_in_files']:
        if entry['pkg'] and len(entry['pkg']) == 1:
            pkgs_matched.append(entry['pkg'][0]['pkg'])
            pkg_name = entry['pkg'][0]['pkg']
            # Remove '.x86_64' or '.i686' from the package name
            if '.' in pkg_name:
                pkg_name = pkg_name[:pkg_name.rfind('.')]
            pkgs[entry['pkg_common'][0]] += 1
            text.append(entry['pkg'][0]['summary'].lower())
        else:
            pkg_name = find_common_pkg_name(entry)
            if pkg_name:
                pkgs[pkg_name] += 1
                text.append(find_common_summary(entry))
    
    pkgs_bin = []
    for entry in data['packages_in_binaries']:
        if entry['comm_path'].endswith('/srun'):
            continue
        pkgs_bin.append(entry['pkg'])
        text.append(entry['summary'].lower())

    # Many pkgs in pkgs_bin are very similar (start with the same 5 characters), keep only 1
    pkgs_prefix = set()
    for i in range(len(pkgs_bin)):
        unique = True
        common_prefix = pkgs_bin[i]
        for j in range(len(pkgs_bin)):
            if i == j:
                continue
            # Check common prefix
            prefix = os.path.commonprefix([common_prefix, pkgs_bin[j]])
            if len(prefix) > 5:
                unique = False
                common_prefix = prefix
        pkgs_prefix.add(common_prefix)
        if unique:
            pkgs[pkgs_bin[i]] += 1
            pkgs_matched.append(pkgs_bin[i])
    for pkg in pkgs_prefix:
        pkgs[pkg] += 1

    text_raw = text
    text = [process_text(t) for t in text]

    return pkgs_matched, pkgs, text, text_raw


def extract_labels(data):
    pkgs_matched, pkgs, text = extract_pkgs_and_text_from_job(data)

    rake.extract_keywords_from_sentences(text)
    rake_keywords = rake.get_ranked_phrases_with_scores()

    bert_keywords = kw_model.extract_keywords('. '.join(text), keyphrase_ngram_range=(1, 5), stop_words='english')

    return {
        'jobid': data['jobid'],
        'label': data['label'],
        'packages': list(set(pkgs)),
        'packages_matched': list(set(pkgs_matched)),
        'keywords_rake': sorted(list(set(rake_keywords)), reverse=True),
        'keywords_bert': bert_keywords
    }


def extract_labels_collectively(jobs_data):
    data = []
    docs = []
    pkg_in_jobs_counter = Counter()
    for job_data in jobs_data:
        pkgs_matches, pkgs, text, text_raw = extract_pkgs_and_text_from_job(job_data)
        data.append({
            'jobid': job_data['jobid'],
            'label': job_data['label'],
            'packages': list(set(pkgs.keys())),
            'packages_matched': list(set(pkgs_matches)),
            'packages_counter': pkgs,
            'text': text,
            'text_raw': text_raw
        })
        docs.append('. '.join(text))
        for pkg in set(pkgs).union(set(pkgs_matches)):
            pkg_in_jobs_counter[pkg] += 1

    data.append({
        'jobid': 'all',
        'label': 'all',
        'packages': sorted(pkg_in_jobs_counter.items(), key=lambda x: x[1], reverse=True)
    })

    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    for id, doc in enumerate(tfidf_matrix):
        feature_index = doc.nonzero()[1]
        tfidf_scores = zip(feature_index, [doc[0, x] for x in feature_index])
        tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        data[id]['keywords_tfidf'] = [tfidf_feature_names[i] for i, _ in tfidf_scores[:10] if tfidf_feature_names[i] not in ['library', 'libraries', 'lib']]

        rake.extract_keywords_from_sentences(data[id]['text'])
        rake_keywords = rake.get_ranked_phrases_with_scores()
        data[id]['keywords_rake'] = sorted(list(set(rake_keywords)), reverse=True)

    return data


def extract_word_clouds(output_dir, data):
    from sklearn.feature_extraction.text import CountVectorizer
    from wordcloud import WordCloud

    def join_bigrams_and_trigrans(text):
        vectorizer = CountVectorizer(ngram_range=(1, 3))
        X = vectorizer.fit_transform([text])
        bigrams = vectorizer.get_feature_names_out()
        return ' '.join(bigrams)
    
    def save_wordcloud(text, output_file, title):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.clf()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        # plt.title(title)
        plt.savefig(output_file)

    label_to_name_map = {
        'pytorch': 'PyTorch',
        'gromacs': 'GROMACS',
        'lammps': 'Lammps',
        'hpcg': 'HPCG',
        'openfoam': 'OpenFOAM'
    }

    for d in data:
        if d['jobid'] == 'all':
            continue
        name = label_to_name_map[d['label'].split('_')[0]]
        text = ' '.join([join_bigrams_and_trigrans(t) for t in d['text'] if len(t) > 1])
        save_wordcloud(text, os.path.join(output_dir, f'wordcloud_{d["jobid"]}_trigrams.png'), f'Keywords for {name} job (using 1-, 2- and 3-grams)')

        text = ' '.join(d['text'])
        save_wordcloud(text, os.path.join(output_dir, f'wordcloud_{d["jobid"]}.png'), f'Keywords for {name} job')

        text = ""
        for pkg, count in d['packages_counter'].items():
            text += f"{pkg} " * count
        save_wordcloud(text, os.path.join(output_dir, f'wordcloud_{d["jobid"]}_pkgs.png'), f'Packages in {name} job')



def cluster_data(output_dir, data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import numpy as np

    tfidf_vectorizer = TfidfVectorizer()
    docs = [' '.join(d['packages'] + d['packages_matched']) for d in data if d['jobid'] != 'all']
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    kmeans = KMeans(n_clusters=5)
    clusters = kmeans.fit_predict(tfidf_matrix)

    # from gensim.models import Word2Vec
    # from sklearn.cluster import KMeans
    # import numpy as np

    # docs = [' '.join(d['packages'] + d['packages_matched']) for d in data if d['jobid'] != 'all']
    # model = Word2Vec([d.split() for d in docs], vector_size=100, window=5, min_count=1, workers=4)

    # def get_vector(tokens):
    #     # Get vectors for all tokens
    #     vectors = [model.wv[word] for word in tokens if word in model.wv]
    #     # Compute the average vector
    #     if vectors:
    #         return np.mean(vectors, axis=0)
    #     else:
    #         # Return a zero vector if no tokens are found in the model
    #         return np.zeros(model.vector_size)
        
    # for d in data:
    #     if d['jobid'] == 'all':
    #         continue
    #     d['vector'] = get_vector(d['packages'] + d['packages_matched'])

    # kmeans = KMeans(n_clusters=5)
    # clusters = kmeans.fit_predict([d['vector'] for d in data if d['jobid'] != 'all'])

    # if doctovec:
        # from gensim.models import Doc2Vec
        # from gensim.models.doc2vec import TaggedDocument
        # import numpy as np

        # documents = [TaggedDocument(doc, [i]) for i, doc in enumerate([d['packages'] + d['packages_matched'] for d in data if d['jobid'] != 'all'])]
        # model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)
        # model.train(documents, total_examples=model.corpus_count, epochs=30)

        # def get_vector(tokens):
        #     return model.infer_vector(tokens)
        
        # for d in data:
        #     if d['jobid'] == 'all':
        #         continue
        #     d['vector'] = get_vector(d['packages'] + d['packages_matched'])

        # from sklearn.cluster import KMeans
        # kmeans = KMeans(n_clusters=5)
        # clusters = kmeans.fit_predict([d['vector'] for d in data if d['jobid'] != 'all'])

    for i, d in enumerate(data):
        if d['jobid'] == 'all':
            continue
        d['cluster'] = int(clusters[i])

    with open(os.path.join(output_dir, 'clusters.json'), 'w') as f:
        cluster_data = [
            {
                'jobid': int(d['jobid']),
                'label': d['label'],
                'cluster': int(d['cluster'])
            } for d in data if d['jobid'] != 'all']
        json.dump(
            sorted(cluster_data, key=lambda x: x['cluster']), f, indent=4)
        
    plt.clf()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(tfidf_matrix.toarray())
    # reduced_vectors = pca.fit_transform([d['vector'] for d in data if d['jobid'] != 'all'])

    # Use unique labels for the legend
    unique_labels = [(d['cluster'], d['label']) for d in data if d['jobid'] != 'all']
    unique_labels = sorted(list(set(unique_labels)))
    
    colors = []
    base_colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i in range(5):
        # count how many samples have cluster i
        count = len([d for d in data if d['jobid'] != 'all' and d['cluster'] == i])
        shades = list(mcolors.LinearSegmentedColormap.from_list("", [base_colors[i], "white"])(np.linspace(0, 1, count + 5)))
        colors.extend(shades[:count])



    for i, x in enumerate(unique_labels):
        _, label = x
        indices = [j for j, d in enumerate(data) if d['label'] == label]
        plt.scatter(reduced_vectors[indices, 0], reduced_vectors[indices, 1], label=label, color=colors[i])

    plt.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.02, 1))
    plt.title('Clusters based on packages')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clusters.png'))


def cluster_on_strings(output_dir, data):
    from gensim.models import Doc2Vec
    from gensim.models.doc2vec import TaggedDocument
    import numpy as np

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate([d['text'] for d in data if d['jobid'] != 'all'])]
    model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)
    model.train(documents, total_examples=model.corpus_count, epochs=30)

    def get_vector(tokens):
        return model.infer_vector(tokens)
    
    for d in data:
        if d['jobid'] == 'all':
            continue
        d['vector'] = get_vector(d['text'])

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5)
    clusters = kmeans.fit_predict([d['vector'] for d in data if d['jobid'] != 'all'])

    for i, d in enumerate(data):
        if d['jobid'] == 'all':
            continue
        d['cluster'] = int(clusters[i])

    with open(os.path.join(output_dir, 'clusters_summaries_doc2vec.json'), 'w') as f:
        cluster_data = [
            {
                'jobid': int(d['jobid']),
                'label': d['label'],
                'cluster': int(d['cluster'])
            } for d in data if d['jobid'] != 'all']
        json.dump(
            sorted(cluster_data, key=lambda x: x['cluster']), f, indent=4)
        
    plt.clf()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform([d['vector'] for d in data if d['jobid'] != 'all'])

    # Use unique labels for the legend
    unique_labels = [(d['cluster'], d['label']) for d in data if d['jobid'] != 'all']
    unique_labels = sorted(list(set(unique_labels)))
    
    colors = []
    base_colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i in range(5):
        # count how many samples have cluster i
        count = len([d for d in data if d['jobid'] != 'all' and d['cluster'] == i])
        shades = list(mcolors.LinearSegmentedColormap.from_list("", [base_colors[i], "white"])(np.linspace(0, 1, count + 5)))
        colors.extend(shades[:count])



    for i, x in enumerate(unique_labels):
        _, label = x
        indices = [j for j, d in enumerate(data) if d['label'] == label]
        plt.scatter(reduced_vectors[indices, 0], reduced_vectors[indices, 1], label=label, color=colors[i])

    plt.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.02, 1))
    plt.title('Clusters based on summaries')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clusters_summaries_doc2vec.png'))



# def cluster_data(output_dir, data):
#     from gensim.models import Word2Vec
#     from sklearn.cluster import KMeans
#     import numpy as np

#     model = Word2Vec([d['text'] for d in data if d['jobid'] != 'all'], vector_size=100, window=5, min_count=1, workers=4)
#     model.train([d['text'] for d in data if d['jobid'] != 'all'], total_examples=len([d['text'] for d in data if d['jobid'] != 'all']), epochs=30)

#     def get_vector(tokens):
#         # Get vectors for all tokens
#         vectors = [model.wv[word] for word in tokens if word in model.wv]
#         # Compute the average vector
#         if vectors:
#             return np.mean(vectors, axis=0)
#         else:
#             # Return a zero vector if no tokens are found in the model
#             return np.zeros(model.vector_size)
        
#     for d in data:
#         if d['jobid'] == 'all':
#             continue
#         d['vector'] = get_vector(d['text'])

#     kmeans = KMeans(n_clusters=5)
#     clusters = kmeans.fit_predict([d['vector'] for d in data if d['jobid'] != 'all'])

#     for i, d in enumerate(data):
#         if d['jobid'] == 'all':
#             continue
#         d['cluster'] = int(clusters[i])

#     with open(os.path.join(output_dir, 'clusters.json'), 'w') as f:
#         cluster_data = [
#             {
#                 'jobid': int(d['jobid']),
#                 'label': d['label'],
#                 'cluster': int(d['cluster'])
#             } for d in data if d['jobid'] != 'all']
#         json.dump(
#             sorted(cluster_data, key=lambda x: x['cluster']), f, indent=4)
        
#     plt.clf()
#     from sklearn.decomposition import PCA

#     pca = PCA(n_components=2)
#     reduced_vectors = pca.fit_transform([d['vector'] for d in data if d['jobid'] != 'all'])

#     # Use unique labels for the legend
#     unique_labels = [d['label'] for d in data if d['jobid'] != 'all']
#     unique_labels = sorted(list(set(unique_labels)))
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
#     for i, label in enumerate(unique_labels):
#         indices = [j for j, d in enumerate(data) if d['label'] == label]
#         plt.scatter(reduced_vectors[indices, 0], reduced_vectors[indices, 1], label=label, color=colors[i])

#     plt.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.02, 1))
#     plt.title('Clusters based on packages')
#     plt.xlabel('PCA 1')
#     plt.ylabel('PCA 2')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'clusters.png'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract labels from a JSON file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=str, help='Path to the JSON file')
    group.add_argument('--dir', type=str, help='Path to the directory containing JSON files')
    args = parser.parse_args()

    if args.file:
        with open(args.file, 'r') as f:
            data = json.load(f)

        data = extract_labels(data)

        output_file = args.file.replace('.json', '_labels.json')
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
    elif args.dir:
        jobs_data = []
        for file in os.listdir(args.dir):
            # Check file name is "job_%d.json"
            if re.match(r'job_\d+\.json', file):
                with open(os.path.join(args.dir, file), 'r') as f:
                    data = json.load(f)
                    jobs_data.append(data)
        
        labels_data = sorted(extract_labels_collectively(jobs_data), key=lambda x: str(x['jobid']))
        # cluster_data(args.dir, labels_data)
        # cluster_on_strings(args.dir, labels_data)
        extract_word_clouds(args.dir, labels_data)
        with open(os.path.join(args.dir, 'labels.json'), 'w') as f:
            json.dump(labels_data, f, indent=4)
