#include <stdio.h>
#include <iostream>
#include <random>
#include <string>
#include <thread>

using namespace std;

const int SAMPLE_SIZE = 92675;
const int HIDDEN_SIZE = 256; // 8 bit (2 ** 8)
const int MELS_DIM    = 80;
const int AUX_DIM     = 32;

const int SHIFT1 = 10;
const int SHIFT2 = 20;
const int BIT1   = 1024;    // 2 ** 10
const int BIT2   = 1048576; // 2 ** 20

int mels[SAMPLE_SIZE][MELS_DIM];
int aux_0[SAMPLE_SIZE][AUX_DIM];
int aux_1[SAMPLE_SIZE][AUX_DIM];
int aux_2[SAMPLE_SIZE][AUX_DIM];
int aux_3[SAMPLE_SIZE][AUX_DIM];

template <class T, const int n> class Array {
    T arr[n] = {};
    T max    = 0;
    T min    = 0;
public:
    void set(const int i, const T v) {
        arr[i] = v;
        if (max < v) max = v;
        if (min > v) min = v;
    }

    T get(const int i) const {
        return arr[i];
    }

    int size() const {
        return n;
    }

    void print(string s) {
        if (typeid(T) == typeid(long)) {
            printf("[%s] max:%ld, min=%ld\n", s.c_str(), max, min);
        } else {
            printf("[%s] max:%d, min=%d\n", s.c_str(), max, min);
        }
    }
};

/**
 * Exponential
 *
 * [input precision] 20 bit
 * [output precision] 20 bit
 */
long exp_d(const int x) {
    double tmp = x;

    tmp /= BIT2;
    tmp  = exp(tmp);
    tmp *= BIT2;

    return tmp;
}

/**
 * Activation relu
 *
 * [input precision] 20 bit
 * [output precision] 20 bit
 */
template <class T> void relu(T &x) {
    for (short i = 0; i < x.size(); ++i) {
        if (0 > x.get(i)) x.set(i, 0);
    }
}

/**
 * Activation sigmoid
 *
 * [input precision] 20 bit
 * [output precision] 10 bit
 */
short sigmoid_d(const int x) {
    int threshold = 2621440; // 2.5 * BIT2

    if (-threshold > x) return 0;
    else if (threshold < x) return BIT1;
    else return x / 5120 + 512; // 0.2x + 0.5
}

/**
 * Activation tanh
 *
 * [input precision] 20 bit
 * [output precision] 20 bit
 */
int tanh_d(const int x) {
    if (-BIT2 > x) return -BIT2;
    else if (BIT2 < x) return BIT2;
    else return x;
}

/**
 * Activation softmax
 *
 * [input precision] 20 bit
 * [output precision] 10 bit
 */
template <class T> void softmax(T &x) {
    int max = 0;

    for (short i = 0; i < x.size(); ++i) {
        if (max < x.get(i)) max = x.get(i);
    }

    long sum = 0;
    long tmp;

    for (short i = 0; i < x.size(); ++i) {
        tmp  = exp_d(x.get(i) - max);
        sum += tmp;
        x.set(i, tmp);
    }

    if (sum == 0) sum = 1;

    for (short i = 0; i < x.size(); ++i) {
        x.set(i, (x.get(i) * BIT2) / sum);
    }
}

/**
 * Categorical distribution
 *
 * [input precision] 10 bit
 */
template <class T> short choice(const T &x) {
    random_device rnd;
    mt19937       mt(rnd());
    uniform_int_distribution <> dist(0, BIT2);
    int threshold = dist(mt);

    for (short i = 0; i < x.size(); ++i) {
        if (threshold < x.get(i)) return i;
        threshold -= x.get(i);
    }

    return 0;
}

template <class T> void add(T &out, const T &in) {
    for (short i = 0; i < out.size(); ++i) {
        out.set(i, out.get(i) + in.get(i));
    }
}

template <class T1, class T2, class T3, int p1, int p2> void concat(T1 &out, const T2 v, const T3 (&in1)[p1], const T3 (&in2)[p2]) {
    out.set(0, v);

    for (short i = 0; i < p1; ++i) {
        out.set(1 + i, in1[i]);
    }

    for (short i = 0; i < p2; ++i) {
        out.set(1 + p1 + i, in2[i]);
    }
}

template <class T1, class T2, class T3, int p2> void concat_2(T1 &out, const T2 &in1, const T3 (&in2)[p2]) {
    short p1 = in1.size();

    for (short i = 0; i < p1; ++i) {
        out.set(i, in1.get(i));
    }

    for (short i = 0; i < p2; ++i) {
        out.set(p1 + i, in2[i]);
    }
}

template <class T> void savefile(T &out) {
    FILE *fp;

    fp = fopen("output.txt", "w");
    for (int i = 0; i < out.size(); ++i) {
        fprintf(fp, "%d\n", out.get(i));
    }

    fclose(fp);
}

void loadfile() {
    // mels
    thread t1([](){
              FILE *fp_mels;

              fp_mels = fopen("inputs/mels.txt", "r");
              for (int i = 0; i < SAMPLE_SIZE; ++i) {
                  for (short j = 0; j < MELS_DIM; ++j) {
                      fscanf(fp_mels, "%d", &mels[i][j]);
                  }
              }
              fclose(fp_mels);
        });

    // aux_0
    thread t2([](){
              FILE *fp_aux_0;

              fp_aux_0 = fopen("inputs/aux_0.txt", "r");
              for (int i = 0; i < SAMPLE_SIZE; ++i) {
                  for (short j = 0; j < AUX_DIM; ++j) {
                      fscanf(fp_aux_0, "%d", &aux_0[i][j]);
                  }
              }
              fclose(fp_aux_0);
        });

    // aux_1
    thread t3([](){
              FILE *fp_aux_1;

              fp_aux_1 = fopen("inputs/aux_1.txt", "r");
              for (int i = 0; i < SAMPLE_SIZE; ++i) {
                  for (short j = 0; j < AUX_DIM; ++j) {
                      fscanf(fp_aux_1, "%d", &aux_1[i][j]);
                  }
              }
              fclose(fp_aux_1);
        });

    // aux_2
    thread t4([](){
              FILE *fp_aux_2;

              fp_aux_2 = fopen("inputs/aux_2.txt", "r");
              for (int i = 0; i < SAMPLE_SIZE; ++i) {
                  for (short j = 0; j < AUX_DIM; ++j) {
                      fscanf(fp_aux_2, "%d", &aux_2[i][j]);
                  }
              }
              fclose(fp_aux_2);
        });

    // aux_3
    thread t5([](){
              FILE *fp_aux_3;

              fp_aux_3 = fopen("inputs/aux_3.txt", "r");
              for (int i = 0; i < SAMPLE_SIZE; ++i) {
                  for (short j = 0; j < AUX_DIM; ++j) {
                      fscanf(fp_aux_3, "%d", &aux_3[i][j]);
                  }
              }
              fclose(fp_aux_3);
        });

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
}
