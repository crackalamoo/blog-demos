import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# NDIM = 50
NDIM = 300

embeddings = {}

with open(f"glove/glove.6B.{NDIM}d.txt", "r") as f:
    glove_content = f.read().split('\n')
    f.close()

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
euc_sim = lambda a,b: -np.sum(np.square(a - b))

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
        if to_word is not None:
            label = to_word.upper()
        else:
            label = ''
    print(label)
    if reverse:
        sims.reverse()
    for i, sim in enumerate(reversed(sims[-n:])):
        print(i+1, display(sim))
    return sims


def get_pca_vecs(n=10):
    pca = PCA()
    X = np.array([embeddings[w] for w in embeddings])
    pca.fit(X)
    principal_components = list(pca.components_[:n, :])
    return pca, principal_components

def get_kmeans_centers(n=600):
    kmeans = KMeans(n_clusters=n, n_init=1)
    X = np.array([embeddings[w] for w in embeddings])
    kmeans.fit(X)
    print(kmeans.n_iter_)
    centers = list(kmeans.cluster_centers_)
    centers.sort(key=lambda v: np.sum(np.square(v)), reverse=True)
    centers = centers[:-100]
    return kmeans, centers

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
        ax.scatter(X, Y, Z, c=colors)
        for i in np.random.choice(len(words), size=100, replace=False):
            ax.text(X[i], Y[i], Z[i], words[i])
    else:
        plt.scatter(X, Y, c=colors)
        for i in np.random.choice(len(words), size=500, replace=False):
            plt.annotate(words[i], (X[i], Y[i]))
    plt.show()

if __name__ == '__main__':
    display_sims(to_word='cat')
    display_sims(to_word='frog')
    display_sims(to_word='red')
    display_sims(to_word='share')
    display_sims(to_word='speak')
    display_sims(to_word='happy')

    display_sims(to_e = embeddings['aunt'] - embeddings['woman'] + embeddings['man'], metric=euc_sim, n=15)
    display_sims(to_e = embeddings['sushi'] - embeddings['japan'] + embeddings['germany'], metric=euc_sim, n=15)
    display_sims(to_e = embeddings['sushi'] - embeddings['japan'] + embeddings['india'], metric=euc_sim, n=15)
    display_sims(to_e = embeddings['sushi'] - embeddings['japan'] + embeddings['france'], metric=euc_sim, n=15)

    zero_vec = np.zeros_like(embeddings['the'])
    display_sims(to_e=zero_vec, metric=euc_sim, reverse=True, label='largest magnitude')
    display_sims(to_e=zero_vec, metric=euc_sim, label='smallest magnitude')

    gender_pairs = [('man', 'woman'), ('men', 'women'), ('brother', 'sister'), ('father', 'mother'),
                    ('uncle', 'aunt'), ('grandfather', 'grandmother'), ('boy', 'girl'),
                    ('son', 'daughter')]
    masc_v = zero_vec
    for pair in gender_pairs:
        masc_v += embeddings[pair[0]]
        masc_v -= embeddings[pair[1]]

    display_sims(to_e=masc_v, metric=cos_sim, label='masculine vecs')
    display_sims(to_e=masc_v, metric=cos_sim, reverse=True, label='feminine vecs')

    pca, pca_vecs = get_pca_vecs()
    for i, vec in enumerate(pca_vecs):
        display_sims(to_e=vec, metric=cos_sim, label=f'PCA {i+1}')
        display_sims(to_e=vec, metric=cos_sim, label=f'PCA {i+1} negative', reverse=True)
    
    # plot_pca(pca_vecs)
    
    kmeans, cluster_centers = get_kmeans_centers()
    for i, vec in enumerate(cluster_centers):
        display_sims(to_e=vec, metric=euc_sim, label=f'KMeans {i+1}')
    
    plot_pca(pca_vecs, kmeans=kmeans)