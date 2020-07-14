'''
    File name: semanticmatching.py
    Author: Hugo Haggren, Leo Hatvani
'''

import pandas as pd
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import hdbscan
import numpy as np
from stringmatching import load_data
import matplotlib.pyplot as plt
from Levenshtein.StringMatcher import StringMatcher
from itertools import combinations
import random

_vector_size = 100
_min_count = 1
_epochs = 100
_seed = 123

def extract_doc2Vec_features(sentences):
    # Completely redundant atm
    # Taken from Auwn
    
    documents = list(sentences)

    # Remove common words and tokenize
    remove_stop_words = False
    if remove_stop_words:
        stoplist = set('for a of the and to in'.split())
        texts = [ [word for word in document.lower().split() if word not in stoplist] for document in documents ]

    # Remove words that appear only once
    remove_uncommon_words = False
    if remove_uncommon_words:
        from collections import defaultdict
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        texts = [[token for token in text if frequency[token] > 1] for text in texts]

    # Apply Doc2Vec model from gensim
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
    model = Doc2Vec(vector_size=_vector_size, min_count=_min_count, epochs=_epochs, seed=_seed)
    model.build_vocab(documents)

    inf_vec = [model.infer_vector(text) for text in texts]
    features = pd.DataFrame(np.row_stack(inf_vec))

    return features

def apply_hdbscan(features):
    import hdbscan
    from sklearn.metrics import pairwise_distances
    distance = pairwise_distances(features, metric='cosine')
    hdb = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed')
    hdb.fit(distance.astype('float64'))


    # Clustering Results
    # Number of clusters in pred_labels, ignoring noise (-1) if present.
    pred_labels = hdb.labels_
    n_clusters_ = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
    n_noise_ = list(pred_labels).count(-1)
    return pred_labels, n_clusters_, n_noise_


def visualize_features(fig_file, embedder):
    # Applying TSNE on results and visualising in scatterplot
    # 
    # In:
    # fig_file - Outfile location
    # embedder - "doc2vec" or "sbert"
    from sklearn.manifold import TSNE
    doc_labels, docs = load_data()
    if embedder == "doc2vec":
        pred_labels, features, dic = perform_clustering()
    if embedder == "sbert":
        pred_labels, features, sbl = sbert_labels()

    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=1000, random_state=123)
    tsne_df = pd.DataFrame(tsne.fit_transform(features))
    tsne_df['testcase_id'] = doc_labels
    tsne_df['labels'] = pred_labels

    # Setting colors for labels
    unique_labels = set(pred_labels)

    print(len(unique_labels))
    #plt.style.use('ggplot')
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels)+1)]
    colors.append('lightgray')
    cl = [colors[i] for i in pred_labels]

    fig = plt.figure()
    ax = plt.subplot(111)
    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    for i,type in enumerate(doc_labels):
        x = tsne_df.iloc[i,0]
        y = tsne_df.iloc[i,1]

        plt.scatter(x, y, s=30, marker='o', color=cl[i], alpha=0.8)
        #plt.text(x+0.3, y+0.3, type, fontsize=6)

    # plt.xlabel('t-SNE1')
    # plt.ylabel('t-SNE2')

    import matplotlib.patches as mpatches
    recs = []
    for i in range(0,len(unique_labels)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))

    # Put a legend to the right of the current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    #ax.legend(recs,unique_labels,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # plt.show()

    fig.savefig(fig_file)

def perform_clustering():
    # Creates doc2vec model and performs HDBSCAN clustering
    #
    # Out:
    # labels - Cluster labels
    # X - feature vectors
    # labeldict - labels in dictionary format
    doc_labels, docs = load_data()

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]

    model = Doc2Vec(documents, vector_size=_vector_size, min_count=_min_count, epochs=_epochs, seed=_seed, workers=1)

    X=[]
    for i in range(len(docs)):
        X.append(model.docvecs[i])
        #print(model.docvecs[i])

    data = np.array(X)
    # print(data.shape)

    #clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
    #cluster_labels = clusterer.fit_predict(data)
    labels, n_clusters, noise = apply_hdbscan(data)

    d = {'col1':doc_labels,'col2':labels}
    df = pd.DataFrame(data=d)
    #df.to_excel('doc2vec_2labels.xlsx')
    dl = [x[:33].strip() for x in doc_labels]
    dictionary = dict(zip(dl, labels))

    return labels, X, dictionary


def load_SME_binary():
    # Loads the labeling from SME
    #
    # Out:
    # SME_labels - labels in dictionary format
    # y_bt - Binary target vector
    df = pd.read_excel(r'test-lab.xlsx')
    dval = df.values
    
    y_true = dval[:,2]
    #y_t = np.delete(y_true, [75, 92])
    y_bt = np.zeros(y_true.shape)

    for i,c in enumerate(y_true):
        if c == 'S':
            y_bt[i] = 1
        else:
            y_bt[i] = 0

    SME_labels = {}
    for i in range(len(dval)):
        doc1 = dval[i,0].strip()
        doc2 = dval[i,1].strip()
        label = dval[i,2]

        if label == 'S':
            if doc1 not in SME_labels:
                SME_labels[doc1] = 1
            if doc2 not in SME_labels:
                SME_labels[doc2] = 1
        else:
            if doc1 not in SME_labels:
                SME_labels[doc1] = 0
            if doc2 not in SME_labels:
                SME_labels[doc2] = 0
    print(len(SME_labels))            
    return SME_labels, y_bt 

def binary_evaluate():
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    doc_labels, docs = load_data()
    labels = perform_clustering()
    y_p = []
    y_t = []
    SME_labels = load_SME()
    for i,dl in enumerate(doc_labels):
        if labels[i] == -1:
            c = 0
        else:
            c = 1
        if dl[:33].strip() in SME_labels:
            y_t.append(SME_labels[dl[:33].strip()])
            y_p.append(c)        


    p, r, f1, supp = precision_recall_fscore_support(y_t, y_p, labels=[0,1])
    acc = accuracy_score(y_t, y_p)
    
    print('prec: ', p)
    print('rec: ', r)
    print('f1: ', f1)
    print('acc: ', acc)

def sbert_labels():
    # Returns labels and sbert-feature vectors for all test cases
    #
    # Out:
    # labels - Cluster labels
    # features - feature vectors
    # labeldict - labels in dictionary format
    import pickle
    df = pd.read_excel(r'excelfiles/sbert_2labels.xlsx')
    vals = df.values

    dl = [x[:33].strip() for x in vals[:,0]]
    labels = list(vals[:,1])
    labeldict = dict(zip(dl, labels))
    infile = open('sbert_features2.pklz','rb')
    features = pickle.load(infile)
    infile.close()

    return labels ,features, labeldict


def evaluate(embedder="doc2vec"):
    # Evaluates the clustering results from HDBSCAN with 
    # SME-labels as ground truth
    #
    # In:
    # embedder - "doc2vec" or "sbert"

    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    doc_labels, docs = load_data()
    if embedder == "doc2vec":
        l, x, labels = perform_clustering()
    elif embedder == "sbert":
        l, x, labels = sbert_labels()
    else:
        return print("wrong input to eval method")
    s,y_t = load_SME_binary()
    df = pd.read_excel(r'test-lab.xlsx')
    dval = df.values
    y_p = []

    # For every pair in the manually determined ground truth, check the 
    # corresponding pair in the automatically derived prediction
    for i in range(len(dval)):
        doc1 = dval[i,0].strip()
        doc1_ = doc1
        for x in labels:
            if x.startswith(doc1):
                doc1_ = x
        doc2 = dval[i,1].strip()
        doc2_ = doc2 
        for x in labels:
            if x.startswith(doc2):
                doc2_ = x
        label = dval[i,2]
        if labels[doc1_] == -1 or labels[doc2_] == -1:
            y_p.append(0)
        else:
            if labels[doc1_] == labels[doc2_]:
                #print(labels[doc1], labels[doc2])
                y_p.append(1)
            else:
                y_p.append(0)

    p, r, f1, supp = precision_recall_fscore_support(y_t, y_p, labels=[0,1])
    acc = accuracy_score(y_t, y_p)
    
    print('prec: ', p)
    print('rec: ', r)
    print('f1: ', f1)
    print('acc: ', acc)
    print(labels)

def compute_levensthein_distances_in_clusters(output_file="", embedder="doc2vec"):
    # Computes the levensthein distances within clusters
    #
    # In:
    # embedder - "doc2vec" or "sbert"

    doc_labels, docs = load_data()
    if embedder == "doc2vec":
        l, x, labels = perform_clustering()
    elif embedder == "sbert":
        l, x, labels = sbert_labels()
    else:
        return print("wrong input to eval method")
    valid_labels = set(l)
    for lbl in valid_labels:
        if lbl==-1:
            continue
        print(lbl,end=",")
        # Create a subset of labels that belong to cluster lbl
        current_cluster = []
        for k,v in labels.items():
            if v == lbl:
                current_cluster.append(k)
        # breakpoint()
        for a,b in combinations(current_cluster, 2):
            idx_a = -1
            idx_b = -1
            for i in range(len(doc_labels)):
                if doc_labels[i].startswith(a):
                    idx_a = i
                elif doc_labels[i].startswith(b):
                    idx_b = i
            doc1 = docs[idx_a]
            doc2 = docs[idx_b]
            # print(doc1,doc2,"a b", a,b)
            # assert(doc1)
            # assert(doc2)
            m = StringMatcher(seq1=doc1, seq2=doc2)
            print(m.ratio(),end=",")
        print()
    return    
    
def compute_levensthein_distances_in_ground_truth():
    # Computes the levensthein distances within similar and non similar pairs of the ground truth
    #
    labels, docs = load_data()
    s,y_t = load_SME_binary()
    df = pd.read_excel(r'test-lab.xlsx')
    dval = df.values
    
    for i in range(len(dval)):
        doc1 = dval[i,0].strip()
        doc1_ = doc1
        for x in labels:
            if x.startswith(doc1):
                doc1_ = x
        doc2 = dval[i,1].strip()
        doc2_ = doc2 
        for x in labels:
            if x.startswith(doc2):
                doc2_ = x
        label = dval[i,2]
        m = StringMatcher(seq1=docs[labels.index(doc1_)], seq2=docs[labels.index(doc2_)])
        print(doc1, doc2, label, m.ratio(), sep=",")

    return  

if __name__=="__main__":
    # while True:
        # _vector_size = random.randint(1,800)
        # _min_count = random.randint(1,5)
        # _epochs = random.randint(1,5)*25
        # _seed = random.randint(1,10000)
    print(_vector_size, _min_count, _epochs, _seed)
    evaluate("sbert")
    # visualize_features('images2/tsne_sbert.pdf', 'sbert')    
    # visualize_features('images2/tsne_doc2vec.pdf', 'doc2vec')
    # compute_levensthein_distances_in_clusters("sbert_clusters.csv", "sbert")
    # compute_levensthein_distances_in_ground_truth()
