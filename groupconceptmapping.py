#! /usr/bin/env python3

############################################################################
#
#   This implements Jackson and Trochim's "Concept Mapping" algorithm [1].
#
#   [1] Kristin M. Jackson and William M. K. Trochim,
#       "Concept Mapping as an Alternative Approach for the Analysis
#       of Open-Ended Survey Responses", Organizational Research Methods,
#       Vol. 5 No. 4, October 2002 (pp. 307-336)
#
############################################################################

import argparse
import collections
import functools
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import AgglomerativeClustering

sns.set(style='white', rc={'axes.edgecolor': 'white'})

# -----------------  DEFINITION OF OPTIONS AND ARGUMENTS  ------------------

parser = argparse.ArgumentParser(
    description="An implementation of Jackson and Trochim's Concept Mapping",
    usage='%(prog)s [--help] [options] input.xlsx min_clusters [max_clusters]',
)
parser.add_argument(
    'filename',
    metavar='input.xlsx',
    help='filename of input data',
)
parser.add_argument(
    'min_clusters',
    help='minimum number of clusters',
    type=int,
)
parser.add_argument(
    'max_clusters',
    help='maximum number of clusters',
    nargs='?',
    type=int,
    default=0,
)

#   By default, we use t-SNE to do the dimension reduction,
#   since it is better at preserving local structure than MDS.
group = parser.add_mutually_exclusive_group()
group.add_argument(
    '--use-tsne',
    help='use t-SNE',
    action='store_true',
    default=True,
)
group.add_argument(
    '--use-mds',
    help='use MDS',
    action='store_false',
    dest='use_tsne',
)

group = parser.add_mutually_exclusive_group()
group.add_argument(
    '--quote-id',
    help='plot the id of each quote',
    action='store_true',
    default=True,
)
group.add_argument(
    '--no-quote-id',
    action='store_false',
    dest='quote_id',
)

group = parser.add_mutually_exclusive_group()
group.add_argument(
    '--label-dot',
    help="plot the location of all proxy sorters's labels",
    action='store_true',
    default=False,
)
group.add_argument(
    '--no-label-dot',
    action='store_false',
    dest='label_dot',
)

group = parser.add_mutually_exclusive_group()
group.add_argument(
    '--label-text',
    help="plot all proxy sorters's labels",
    action='store_true',
    default=False,
)
group.add_argument(
    '--no-label-text',
    action='store_false',
    dest='label_text',
)

group = parser.add_mutually_exclusive_group()
group.add_argument(
    '--cluster-center',
    help='plot the center of each cluster',
    action='store_true',
    default=True,
)
group.add_argument(
    '--no-cluster-center',
    action='store_false',
    dest='cluster_center',
)

group = parser.add_mutually_exclusive_group()
group.add_argument(
    '--cluster-outline',
    help='plot the outline of each cluster',
    action='store_true',
    default=True,
)
group.add_argument(
    '--no-cluster-outline',
    action='store_false',
    dest='cluster_outline',
)

group = parser.add_mutually_exclusive_group()
group.add_argument(
    '--cluster-id',
    help='plot the id of each cluster',
    action='store_true',
    default=False,
)
group.add_argument(
    '--no-cluster-id',
    action='store_false',
    dest='cluster_id',
)

parser.add_argument(
    '--seed',
    help='seed for the random number generator',
    type=int,
)

parser.add_argument(
    '--embed-only',
    help='compute only embedding, not clusters',
    action='store_true',
    default=False,
)


options = parser.parse_args()
if options.max_clusters < options.min_clusters:
    options.max_clusters = options.min_clusters

if options.seed is None:
    options.seed = np.random.get_state()[1][0]
np.random.seed(options.seed)
print('Seed for PRNG:', options.seed)


#   ----    end op option/argument parsing  ----

# colors = plt.cm.get_cmap('tab20')
colors = mpl.colors.ListedColormap([
    '#42145f', '#01689b', '#007bc7', '#a90061', '#ca005d', '#d52b1e',
    '#275937', '#39870c', '#777c00', '#673327', '#94710a', '#f9e11e',
    '#ffb612', '#e17000', '#f092cd', '#8fcae7', '#76d2b6'
])

#   Extract groups (label plus list of quote numbers) from input.
xlsx = pd.read_excel(options.filename, sheet_name=None, header=None, dtype=None)
sheets = list(xlsx.keys())
assert len(sheets) == 1

cardSort = xlsx[sheets[0]]
statements, *groups = [cardSort[i] for i in cardSort]
statements = [s.encode("cp1252").decode("utf_8") for s in statements[1:]]
n_statements = len(statements)

labelled_group = {}
similarity = np.zeros((n_statements, n_statements), dtype=int)
for group in groups:
    label = group[0].encode("cp1252").decode("utf_8")
    indices = [i - 1 for i in range(1, len(group)) if group[i] == 1]
    if len(indices) == 1:
        continue

    labelled_group[label] = indices
    for i, si in enumerate(indices):
        similarity[si][si] += 1
        for sj in indices[i+1:]:
            similarity[si][sj] += 1
            similarity[sj][si] += 1

n_sorters = max(similarity[i][i] for i in range(n_statements))
for i in range(n_statements):
    similarity[i][i] = n_sorters
dissimilarity = pd.DataFrame(1 - similarity/n_sorters)
dissimilarity.to_csv('distances.csv', header=False, index=False)


if options.use_tsne:
    tsne = TSNE(metric='precomputed', init='random', method='exact')
    tsne.fit(dissimilarity)
    embedding = tsne.embedding_
    print('KL-divergence =', tsne.kl_divergence_)
else:
    mds = MDS(n_components=2, metric=False, dissimilarity='precomputed')
    mds.fit(dissimilarity)
    embedding = mds.embedding_
    print('Stress =', mds.stress_)


#   ----    end of embedding, start of clustering   ----

if options.embed_only:
    sys.exit()


def average_point(points):
    '''Helper function to compute the average of an array of points.'''
    return functools.reduce(np.add, points) / len(points)


def squared_distance(a, b):
    '''Helper function to compute the (squared) distance between two points.'''
    return np.dot(a - b, a - b)


#   Compute clustering and lists of quotes per cluster.
for n_clusters in range(options.min_clusters, options.max_clusters + 1):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    clustering.fit(embedding)
    cluster_index = clustering.labels_
    clusters = [None] * n_clusters
    for i in range(n_clusters):
        clusters[i] = []
    for i in range(embedding.shape[0]):
        clusters[cluster_index[i]].append(i)
    clusters.sort(key = lambda cluster: cluster[0])
    for i, cluster in enumerate(clusters):
        for j in cluster:
            cluster_index[j] = i

    #   Plot location of quotes, coloured according to cluster.
    fig, ax = plt.subplots(figsize=[10.80, 10.80], dpi=300)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors(cluster_index[:]),
    )

    #   Plot quote numbers next to dots; note that quote numbers are one-based
    #   whereas Python's indices are zero-based, hence the "+ 1".
    if options.quote_id:
        for i in range(embedding.shape[0]):
            ax.text(embedding[i, 0], embedding[i, 1], i + 1)

    #   Compute and plot the centers of the clusters; plot the convex hull
    #   of the points in each cluster to better show the clusters.
    cluster_centers = [None] * n_clusters
    for i in range(n_clusters):
        points = embedding[clusters[i]]
        avg = average_point(points)
        cluster_centers[i] = avg

        if options.cluster_center:
            ax.scatter(*avg, marker="x", color=colors(i))
        if options.cluster_id:
            avg = [c + .5 for c in avg]
            ax.text(*avg, clusters[i][0] + 1, fontsize=16)

        #   We need a try/except construct because ConvexHull will throw
        #   an Exception if a cluster has less than three points.
        if options.cluster_outline:
            try:
                hull = ConvexHull(points)
                poly = plt.Polygon(
                    points[hull.vertices],
                    fill=True,
                    facecolor=colors(i),
                    alpha=0.25,
                )
                ax.add_patch(poly)
            except Exception:
                pass

    #   Compute and plot the location of all labels given by the proxies.
    location_for_label = {}
    for label, quotes in labelled_group.items():
        #   "- 1" to convert quote #'s (one-based) to indices (zero-based).
        quotes = [q - 1 for q in quotes]
        avg = average_point(embedding[quotes])
        location_for_label[label] = avg
        if options.label_dot:
            ax.scatter(*avg, marker=".", color="#b4b4b4")
        if options.label_text:
            ax.text(*avg, label, horizontalalignment='center',
                    verticalalignment='center')

    fig.savefig('clusters_{}.svg'.format(n_clusters))
    plt.close(fig)

    # For each cluster, print the statements and ten labels closest to its center.
    with open('clusters_{}.txt'.format(n_clusters), 'wt') as clusters_file:
        for i in range(n_clusters):
            print(f"\n============\n CLUSTER {i+1:2d}\n============\n",
                  file=clusters_file)
            print("    ------------\n     Statements\n    ------------",
                  file=clusters_file)
            for j in clusters[i]:
                print(f"        {statements[j]}", file=clusters_file)

            center = cluster_centers[i]
            labels = sorted(location_for_label.keys(),
                            key=lambda q: squared_distance(location_for_label[q],
                                                           center)
                           )[:10]
            print("\n    --------\n     Labels\n    --------",
                  file=clusters_file)
            for label in labels:
                print(f"        {label}", file=clusters_file)
            print(file=clusters_file)

