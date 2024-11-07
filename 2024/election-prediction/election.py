import time
import random
import sys

TRIALS = 1e6

elections = [
    [1789, ["Washington"]],
    [1792, ["Washington"]],
    [1796, ["Adams", "Jefferson"]],
    [1800, ["Jefferson", "Adams"]],
    [1804, ["Jefferson", "Cotesworth"]],
    [1808, ["Madison", "Cotesworth"]],
    [1812, ["Madison", "Clinton"]],
    [1816, ["Monroe", "King"]],
    [1820, ["Monroe"]],
    [1824, ["Adams", "Jackson"]],
    [1828, ["Jackson", "Adams"]],
    [1832, ["Jackson", "Clay"]],
    [1836, ["Van Buren", "Harrison"]],
    [1840, ["Harrison", "Van Buren"]],
    [1844, ["Polk", "Clay"]],
    [1848, ["Taylor", "Cass"]],
    [1852, ["Pierce", "Scott"]],
    [1856, ["Buchanan", "Frémont", "Filmore"]],
    [1860, ["Lincoln", "Breckinridge", "Bell", "Douglas"]],
    [1864, ["Lincoln", "McClellan"]],
    [1868, ["Grant", "Seymour"]],
    [1872, ["Grant", "Greeley"]],
    [1876, ["Hayes", "Tilden"]],
    [1880, ["Garfield", "Hancock"]],
    [1884, ["Cleveland", "Blaine"]],
    [1888, ["Harrison", "Cleveland"]],
    [1892, ["Cleveland", "Harrison", "Weaver"]],
    [1896, ["McKinley", "Bryan"]],
    [1900, ["McKinley", "Bryan"]],
    [1904, ["Roosevelt", "Parker"]],
    [1908, ["Taft", "Bryan"]],
    [1912, ["Wilson", "Roosevelt", "Taft"]],
    [1916, ["Wilson", "Hughes"]],
    [1920, ["Harding", "Cox"]],
    [1924, ["Coolidge", "Davis", "La Follette"]],
    [1928, ["Hoover", "Smith"]],
    [1932, ["Roosevelt", "Hoover"]],
    [1936, ["Roosevelt", "Landon"]],
    [1940, ["Roosevelt", "Wilkie"]],
    [1944, ["Roosevelt", "Dewey"]],
    [1948, ["Truman", "Dewey", "Thurmond"]],
    [1952, ["Eisenhower", "Stevenson"]],
    [1956, ["Eisenhower", "Stevenson"]],
    [1960, ["Kennedy", "Nixon"]],
    [1964, ["Johnson", "Goldwater"]],
    [1968, ["Nixon", "Humphrey", "Wallace"]],
    [1972, ["Nixon", "McGovern"]],
    [1976, ["Carter", "Ford"]],
    [1980, ["Reagan", "Carter"]],
    [1984, ["Reagan", "Ferraro"]],
    [1988, ["Bush", "Dukakis"]],
    [1992, ["Clinton", "Bush"]],
    [1996, ["Clinton", "Dole"]],
    [2000, ["Bush", "Gore"]],
    [2004, ["Bush", "Kerry"]],
    [2008, ["Obama", "McCain"]],
    [2012, ["Obama", "Romney"]],
    [2016, ["Trump", "Clinton"]],
    [2020, ["Biden", "Trump"]],
    [2024, ["Trump", "Harris"]]
]

for e in elections:
    sorted_names = sorted(e[1])
    result = sorted_names.index(e[1][0])
    e.append(len(sorted_names))
    e.append(result)

start = time.time()
max_correct = 0
best_seed = -1

# used to get array data for C++
for e in elections:
    print(e[2], end=', ')
sys.stdout.flush()
print()
for e in elections:
    print(e[3], end=', ')
sys.stdout.flush()
print()

elections = elections[32:]

def simulate_elections(seed):
    random.seed(seed)
    correct = 0
    for j in range(len(elections)):
        result = random.randint(0, elections[j][2]-1)
        if result == elections[j][3]:
            correct += 1
    return correct

for i in range(int(TRIALS)):
    correct = simulate_elections(i)
    if correct >= max_correct:
        max_correct = correct
        best_seed = i
print(time.time() - start)
print(best_seed)
print(f"{max_correct}/{len(elections)}")

correct = 0
for j in range(len(elections)):
    result = random.randint(0, elections[j][2]-1)
    true = elections[j][3]
    sorted_names = sorted(elections[j][1])
    print(f"{elections[j][0]}: {sorted_names[true]}/{sorted_names[result]}")
