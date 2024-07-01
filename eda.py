import numpy as np

embeddings = {}

with open("glove/glove.6B.300d.txt", "r") as f:
    glove_content = f.read().split('\n')
    f.close()

    for i in range(len(glove_content)//4):
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

def display_sims(to_word=None, to_e=None, metric=cos_sim, reverse=False):
    assert (to_word is not None) ^ (to_e is not None)
    sims = get_sims(to_word=to_word, to_e=to_e, metric=metric)
    display = lambda sim: f'{sim[1]}: {sim[0]:.5f}'
    if to_word is not None:
        print(to_word.upper())
    else:
        print()
    if reverse:
        sims.reverse()
    for sim in sims[-10:]:
        print(display(sim))

display_sims(to_word='frog')
display_sims(to_word='share')
display_sims(to_word='speak')
display_sims(to_word='happy')
display_sims(to_word='red')

display_sims(to_e = embeddings['aunt'] - embeddings['woman'] + embeddings['man'], metric=euc_sim)
display_sims(to_e = embeddings['sushi'] - embeddings['japan'] + embeddings['germany'], metric=euc_sim)

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

display_sims(to_e=masc_v, metric=cos_sim)
display_sims(to_e=masc_v, metric=cos_sim, reverse=True)