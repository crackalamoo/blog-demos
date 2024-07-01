import numpy as np
from sklearn.decomposition import PCA

embeddings = {}

with open("glove/glove.6B.300d.txt", "r") as f:
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


def get_pca_vecs(n=10):
    pca = PCA()
    X = np.array([embeddings[w] for w in embeddings])
    pca.fit(X)
    principal_components = list(pca.components_[:n, :])
    return principal_components


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
    display_sims(to_e=zero_vec, metric=euc_sim, reverse=True)
    display_sims(to_e=zero_vec, metric=euc_sim)

    gender_pairs = [('man', 'woman'), ('men', 'women'), ('brother', 'sister'), ('father', 'mother'),
                    ('uncle', 'aunt'), ('grandfather', 'grandmother'), ('boy', 'girl'),
                    ('son', 'daughter')]
    masc_v = zero_vec
    for pair in gender_pairs:
        masc_v += embeddings[pair[0]]
        masc_v -= embeddings[pair[1]]

    display_sims(to_e=masc_v, metric=cos_sim, label='masculine vecs')
    display_sims(to_e=masc_v, metric=cos_sim, reverse=True, label='feminine vecs')

    pca_vecs = get_pca_vecs()
    for i, vec in enumerate(pca_vecs):
        display_sims(to_e=vec, metric=cos_sim, label=f'PCA {i+1}')
        display_sims(to_e=vec, metric=cos_sim, label=f'PCA {i+1} negative', reverse=True)