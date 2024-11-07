#include <iostream>
#include <thread>
#include <vector>
#include <stdint.h>
#include <stdlib.h>
#include <ctime>
using namespace std;

struct SimParams {
    unsigned int seed;
    unsigned int i;
    int n_elections;
    uint8_t *n_candidates;
    uint8_t *results;
    unique_ptr<uint8_t[]>& correct;
    int seeds_per_thread;
};

void simulate_elections(SimParams params) {
    for (int i = 0; i < params.seeds_per_thread; i++) {
        unsigned int r = params.seed + i;
        int correct = 0;
        for (int i = 0; i < params.n_elections; i++) {
            r = rand_r(&r);
            int guess = r % params.n_candidates[i];
            correct += (guess == params.results[i]);
        }
        params.correct[params.i + i] = correct;
    }
}

int main() {
    const int TRIALS = 8000000;
    const int SEEDS_PER_THREAD = 3000;
    const int REPEATS = 400;
    const int SKIP = 32;
    uint8_t n_candidates[] = {1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    uint8_t results[] = {0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 3, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 0, 0, 1, 1, 0, 1, 2, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1};
    int n_elections = sizeof(results) / sizeof(results[0]) - SKIP;

    unsigned int max_correct = 0;
    unsigned int best_seed = 0;

    std::time_t start = std::time(nullptr);
    for (unsigned int e = 0; e < REPEATS; e++) {
        cout << "Repeat: " << e << endl;
        unique_ptr<uint8_t[]> correct_v(new uint8_t[TRIALS]());
        vector<thread> threads = {};
        for (unsigned int i = 0; i < TRIALS; i += SEEDS_PER_THREAD) {
            unsigned int seed = i + TRIALS*e;
            SimParams params = {
                seed, i, n_elections, n_candidates+SKIP, results+SKIP, correct_v, SEEDS_PER_THREAD
            };
            thread t(simulate_elections, params);
            threads.push_back(std::move(t));
        }

        for (unsigned int i = 0; i < threads.size(); i++) {
            threads.at(i).join();
        }
        threads.clear();

        for (unsigned int i = 0; i < TRIALS; i++) {
            unsigned int seed = i + TRIALS*e;
            unsigned int correct = correct_v[i];
            if (correct >= max_correct) {
                max_correct = correct;
                best_seed = seed;
                cout << "Max correct: " << max_correct << endl;
            }
        }

        if (max_correct == n_elections) {
            break;
        }
    }

    cout << "Best seed: " << best_seed << endl;
    cout << "Correct: " << max_correct << "/" << n_elections << endl;
    std::time_t end = std::time(nullptr);
    cout << "Time: " << end - start << "s" << endl;

    srand(best_seed);
    int correct = 0;
    for (int i = 0; i < n_elections; i++) {
        int guess = rand() % (n_candidates[i + SKIP]);
        cout << "Guess: " << guess << ", Actual: " << (int)(results[i + SKIP]) << endl;
        correct += (guess == results[i + SKIP]);
    }
    cout << "Correct: " << correct << "/" << n_elections << endl;
}