import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# NDIM = 50
NDIM = 300

embeddings = {}

with open(f"glove.6B/glove.6B.{NDIM}d.txt", "r") as f:
    glove_content = f.read().split('\n')

for i in range(len(glove_content)//10):
    line = glove_content[i].strip().split(' ')
    assert len(line) != 0
    if line[0] == '':
        continue
    word = line[0]
    embedding = np.array(list(map(float, line[1:])))
    embeddings[word] = embedding

print(len(embeddings))

cos_sim = lambda a,b: np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))
euc_dist = lambda a,b: np.sum(np.square(a - b))

def get_sims(to_word=None, to_e=None, metric=cos_sim):
    assert (to_word is not None) ^ (to_e is not None)
    sims = []
    if to_e is None:
        to_e = embeddings[to_word]
    for word in embeddings:
        if word == to_word:
            continue
        word_e = embeddings[word]
        sim = metric(word_e, to_e)
        sims.append((sim, word))
    sims.sort()
    return sims

def display_sims(to_word=None, to_e=None, n=10, metric=cos_sim, reverse=False, label=None):
    assert (to_word is not None) ^ (to_e is not None)
    sims = get_sims(to_word=to_word, to_e=to_e, metric=metric)
    display = lambda sim: f'{sim[1]}: {sim[0]:.5f}'
    if label is None:
        label = to_word.upper() if to_word is not None else ''
    print(label)
    if reverse:
        sims.reverse()
    for i, sim in enumerate(reversed(sims[-n:])):
        print(i+1, display(sim))


def get_pca_vecs(n=10):
    pca = PCA()
    X = np.array([embeddings[w] for w in embeddings])
    pca.fit(X)
    principal_components = list(pca.components_[:n, :])
    return pca, principal_components

def get_kmeans(n=300):
    kmeans = KMeans(n_clusters=n, n_init=1)
    X = np.array([embeddings[w] for w in embeddings])
    kmeans.fit(X)
    return kmeans

def display_kmeans(kmeans):
    words = np.array([w for w in embeddings])
    X = np.array([embeddings[w] for w in embeddings])
    y = kmeans.predict(X)
    for cluster in range(kmeans.cluster_centers_.shape[0]):
        print(f'KMeans {cluster}')
        cluster_words = words[y == cluster]
        for i, w in enumerate(cluster_words[:5]):
            print(i+1, w)

def get_kmeans_cluster(kmeans, word=None, cluster=None):
    assert (word is None) ^ (cluster is None)
    if cluster is None:
        cluster = kmeans.predict([embeddings[word]])[0]
    words = np.array([w for w in embeddings])
    X = np.array([embeddings[w] for w in embeddings])
    y = kmeans.predict(X)
    cluster_words = words[y == cluster]
    return cluster, cluster_words

def display_cluster(kmeans, word):
    cluster, cluster_words = get_kmeans_cluster(kmeans, word=word)
    print(f"Full KMeans ({word}, cluster {cluster})")
    for i, w in enumerate(cluster_words):
        print(i+1, w)
    distances = np.concatenate([kmeans.cluster_centers_[:cluster], kmeans.cluster_centers_[cluster+1:]], axis=0)
    distances = np.sum(np.square(distances - kmeans.cluster_centers_[cluster]), axis=1)
    nearest = np.argmin(distances, axis=0)
    _, nearest_words = get_kmeans_cluster(kmeans, cluster=nearest)
    print(f"Nearest cluster: {nearest}")
    for i, w in enumerate(nearest_words[:5]):
        print(i+1, w)
    farthest = np.argmax(distances, axis=0)
    print(f"Farthest cluster: {farthest}")
    _, farthest_words = get_kmeans_cluster(kmeans, cluster=farthest)
    for i, w in enumerate(farthest_words[:5]):
        print(i+1, w)

def display_clusters_overview(kmeans):
    X = np.array([embeddings[w] for w in embeddings])
    y = kmeans.predict(X)
    unique, counts = np.unique(y, return_counts=True)
    biggest = unique[np.argmax(counts)]
    smallest = unique[np.argmin(counts)]
    _, biggest_words = get_kmeans_cluster(kmeans, cluster=biggest)
    print("OVERVIEW OF CLUSTERS")
    print(f"Biggest cluster: {biggest} ({len(biggest_words)} words)")
    for i, w in enumerate(biggest_words[:10]):
        print(i+1, w)
    _, smallest_words = get_kmeans_cluster(kmeans, cluster=smallest)
    print(f"Smallest cluster: {smallest} ({len(smallest_words)} words)")
    for i, w in enumerate(smallest_words[:10]):
        print(i+1, w)
    distances = np.zeros(kmeans.cluster_centers_.shape[0])
    for i in range(distances.shape[0]):
        other_centers = np.concatenate([kmeans.cluster_centers_[:i], kmeans.cluster_centers_[i+1:]], axis=0)
        distances[i] = np.mean(np.sum(np.square(other_centers - kmeans.cluster_centers_[i]), axis=1))
    most_isolated = np.argmax(distances)
    _, most_isolated_words = get_kmeans_cluster(kmeans, cluster=most_isolated)
    print(f"Most isolated cluster: {most_isolated}")
    for i, w in enumerate(most_isolated_words[:10]):
        print(i+1, w)
    least_isolated = np.argmin(distances)
    _, least_isolated_words = get_kmeans_cluster(kmeans, cluster=least_isolated)
    print(f"Least isolated cluster: {least_isolated}")
    for i, w in enumerate(least_isolated_words[:10]):
        print(i+1, w)

def display_covariance():
    X = np.array([embeddings[w] for w in embeddings]).T # rows are variables, columns are observations
    cov = np.cov(X)
    cov *= (1 - np.eye(cov.shape[0]))
    cov_range = np.maximum(np.max(cov), np.abs(np.min(cov)))
    plt.imshow(cov, cmap='bwr', interpolation='nearest', vmin=-cov_range, vmax=cov_range)
    plt.colorbar()
    plt.show()


def plot_pca(pca_vecs, plot_3d=False, kmeans=None):
    words = [w for w in embeddings]
    x_vec = pca_vecs[0]
    y_vec = pca_vecs[1]
    X = np.array([np.dot(x_vec, embeddings[w]) for w in words])
    Y = np.array([np.dot(y_vec, embeddings[w]) for w in words])
    colors =  kmeans.predict([embeddings[w] for w in words])
    if plot_3d:
        z_vec = pca_vecs[2]
        Z = np.array([np.dot(z_vec, embeddings[w]) for w in words])
        ax = plt.subplot(projection='3d')
        ax.scatter(X, Y, Z, c=colors, cmap='plasma')
        for i in np.random.choice(len(words), size=100, replace=False):
            ax.text(X[i], Y[i], Z[i], words[i])
    else:
        plt.scatter(X, Y, c=colors, cmap='spring')
        for i in np.random.choice(len(words), size=100, replace=False):
            plt.annotate(words[i], (X[i], Y[i]), weight='bold')
    plt.show()

def plot_magnitudes():
    words = [w for w in embeddings]
    magnitude = lambda word: np.linalg.norm(embeddings[word])
    magnitudes = list(map(magnitude, words))
    plt.hist(magnitudes, bins=40)
    plt.show()

if __name__ == '__main__':
    display_sims(to_word='cat')
    display_sims(to_word='frog')
    display_sims(to_word='red')
    display_sims(to_word='share')
    display_sims(to_word='speak')
    display_sims(to_word='happy')
    display_sims(to_word='queen')

    display_sims(to_e = embeddings['man'] - embeddings['woman'] + embeddings['queen'], metric=cos_sim, n=15, reverse=False, label='king - queen')
    display_sims(to_e = embeddings['aunt'] - embeddings['woman'] + embeddings['man'], metric=cos_sim, n=15, reverse=False, label='aunt - uncle')
    display_sims(to_e = embeddings['sushi'] - embeddings['japan'] + embeddings['germany'], metric=euc_dist, n=15, reverse=True)
    display_sims(to_e = embeddings['sushi'] - embeddings['japan'] + embeddings['india'], metric=euc_dist, n=15, reverse=True)
    display_sims(to_e = embeddings['sushi'] - embeddings['japan'] + embeddings['france'], metric=euc_dist, n=15, reverse=True)

    zero_vec = np.zeros_like(embeddings['the'])
    display_sims(to_e=zero_vec, metric=euc_dist, label='largest magnitude')
    display_sims(to_e=zero_vec, metric=euc_dist, reverse=True, label='smallest magnitude')
    
    # plot_magnitudes()

    gender_pairs = [('man', 'woman'), ('men', 'women'), ('brother', 'sister'), ('he', 'she'),
                    ('uncle', 'aunt'), ('grandfather', 'grandmother'), ('boy', 'girl'),
                    ('son', 'daughter')]
    masc_v = zero_vec
    for pair in gender_pairs:
        masc_v += embeddings[pair[0]]
        masc_v -= embeddings[pair[1]]
    masc_v /= len(gender_pairs)

    display_sims(to_e=masc_v, metric=cos_sim, label='masculine vecs')
    display_sims(to_e=masc_v, metric=cos_sim, reverse=True, label='feminine vecs')
    print("nurse - man", cos_sim(embeddings['nurse'], embeddings['man']))
    print("nurse - woman", cos_sim(embeddings['nurse'], embeddings['woman']))

    pca, pca_vecs = get_pca_vecs()
    for i, vec in enumerate(pca_vecs):
        display_sims(to_e=vec, metric=cos_sim, label=f'PCA {i+1}')
        display_sims(to_e=vec, metric=cos_sim, label=f'PCA {i+1} negative', reverse=True)
    
    # plot_pca(pca_vecs)
    
    kmeans = get_kmeans()
    display_kmeans(kmeans)
    
    plot_pca(pca_vecs, kmeans=kmeans)

    display_cluster(kmeans, 'outbreak')
    display_cluster(kmeans, 'animal')
    display_cluster(kmeans, 'illinois')
    # display_cluster(kmeans, 'maxwell')
    display_cluster(kmeans, 'genghis')

    display_clusters_overview(kmeans)

    display_covariance()

    e9 = np.zeros_like(zero_vec)
    e9[9] = 1.0
    e276 = np.zeros_like(zero_vec)
    e276[276] = 1.0
    display_sims(to_e=e9, metric=cos_sim, label='e9', reverse=True)
    display_sims(to_e=e276, metric=cos_sim, label='e276', reverse=True)