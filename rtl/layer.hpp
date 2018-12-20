#include "function.hpp"

#include "params/fc1_b.txt"
#include "params/fc1_w.txt"
#include "params/fc2_b.txt"
#include "params/fc2_w.txt"
#include "params/fc3_b.txt"
#include "params/fc3_w.txt"
#include "params/I_b.txt"
#include "params/I_w.txt"
#include "params/rnn1_bi.txt"
#include "params/rnn1_bh.txt"
#include "params/rnn1_wi.txt"
#include "params/rnn1_wh.txt"
#include "params/rnn2_bi.txt"
#include "params/rnn2_bh.txt"
#include "params/rnn2_wi.txt"
#include "params/rnn2_wh.txt"

using namespace std;

Array <int, HIDDEN_SIZE * 3> igates;
Array <int, HIDDEN_SIZE * 3> hgates;
Array <short, HIDDEN_SIZE>   reset_gate;
Array <short, HIDDEN_SIZE>   input_gate;
Array <int, HIDDEN_SIZE>     new_gate;

/**
 * Layer Linear_I
 *
 * [input precision] 20 bit
 * [output precision] 20 bit
 */
template <class T1, class T2> void linear_I(T1 &out, const T2 &in) {
    long sum;

    for (short i = 0; i < out.size(); ++i) {
        sum = 0;
        for (short j = 0; j < in.size(); ++j) {
            sum += (in.get(j) / BIT1) * I_w[i][j];
        }
        out.set(i, sum + I_b[i]);
    }
}

/**
 * Layer Linear_fc1
 *
 * [input precision] 20 bit
 * [output precision] 20 bit
 */
template <class T1, class T2> void linear_fc1(T1 &out, const T2 &in) {
    long sum;

    for (short i = 0; i < out.size(); ++i) {
        sum = 0;
        for (short j = 0; j < in.size(); ++j) {
            sum += (in.get(j) / BIT1) * fc1_w[i][j];
        }
        out.set(i, sum + fc1_b[i]);
    }
}

/**
 * Layer Linear_fc2
 *
 * [input precision] 20 bit
 * [output precision] 20 bit
 */
template <class T1, class T2> void linear_fc2(T1 &out, const T2 &in) {
    long sum;

    for (short i = 0; i < out.size(); ++i) {
        sum = 0;
        for (short j = 0; j < in.size(); ++j) {
            sum += (in.get(j) / BIT1) * fc2_w[i][j];
        }
        out.set(i, sum + fc2_b[i]);
    }
}

/**
 * Layer Linear_fc3
 *
 * [input precision] 20 bit
 * [output precision] 20 bit
 */
template <class T1, class T2> void linear_fc3(T1 &out, const T2 &in) {
    long sum;

    for (short i = 0; i < out.size(); ++i) {
        sum = 0;
        for (short j = 0; j < in.size(); ++j) {
            sum += (in.get(j) / BIT1) * fc3_w[i][j];
        }
        out.set(i, sum + fc3_b[i]);
    }
}

/**
 * Layer Linear_rnn1_i
 *
 * [input precision] 20 bit
 * [output precision] 20 bit
 */
template <class T1, class T2> void linear_rnn1_i(T1 &out, const T2 &in) {
    long sum;

    for (short i = 0; i < out.size(); ++i) {
        sum = 0;
        for (short j = 0; j < in.size(); ++j) {
            sum += (in.get(j) / BIT1) * rnn1_wi[i][j];
        }
        out.set(i, sum + rnn1_bi[i]);
    }
}

/**
 * Layer Linear_rnn1_h
 *
 * [input precision] 20 bit
 * [output precision] 20 bit
 */
template <class T1, class T2> void linear_rnn1_h(T1 &out, const T2 &in) {
    long sum;

    for (short i = 0; i < out.size(); ++i) {
        sum = 0;
        for (short j = 0; j < in.size(); ++j) {
            sum += (in.get(j) / BIT1) * rnn1_wh[i][j];
        }
        out.set(i, sum + rnn1_bh[i]);
    }
}

/**
 * Layer RNN_1
 *
 * [input precision] 20 bit
 * [output precision] 20 bit
 */
template <class T1, class T2> void rnn_1(const T1 &x, T2 &h) {
    // igates, hgates
    linear_rnn1_i(igates, x);
    linear_rnn1_h(hgates, h);

    // reset_gate, input_gate, new_gate
    short h_size = h.size();
    for (short i = 0; i < h_size; ++i) {
        reset_gate.set(i, sigmoid_d(igates.get(i) + hgates.get(i)));
        input_gate.set(i, sigmoid_d(igates.get(h_size + i) + hgates.get(h_size + i)));
        new_gate.set(i, tanh_d(igates.get(h_size * 2 + i) + reset_gate.get(i) * (hgates.get(h_size * 2 + i) / BIT1)));

        // h_next
        h.set(i, new_gate.get(i) + input_gate.get(i) * ((h.get(i) - new_gate.get(i)) / BIT1));
    }
}

/**
 * Layer Linear_rnn2_i
 *
 * [input precision] 20 bit
 * [output precision] 20 bit
 */
template <class T1, class T2> void linear_rnn2_i(T1 &out, const T2 &in) {
    long sum;

    for (short i = 0; i < out.size(); ++i) {
        sum = 0;
        for (short j = 0; j < in.size(); ++j) {
            sum += (in.get(j) / BIT1) * rnn2_wi[i][j];
        }
        out.set(i, sum + rnn2_bi[i]);
    }
}

/**
 * Layer Linear_rnn2_h
 *
 * [input precision] 20 bit
 * [output precision] 20 bit
 */
template <class T1, class T2> void linear_rnn2_h(T1 &out, const T2 &in) {
    long sum;

    for (short i = 0; i < out.size(); ++i) {
        sum = 0;
        for (short j = 0; j < in.size(); ++j) {
            sum += (in.get(j) / BIT1) * rnn2_wh[i][j];
        }
        out.set(i, sum + rnn2_bh[i]);
    }
}

/**
 * Layer RNN_2
 *
 * [input precision] 20 bit
 * [output precision] 20 bit
 */
template <class T1, class T2> void rnn_2(const T1 &x, T2 &h) {
    // igates, hgates
    linear_rnn2_i(igates, x);
    linear_rnn2_h(hgates, h);

    // reset_gate, input_gate, new_gate
    short h_size = h.size();
    for (short i = 0; i < h_size; ++i) {
        reset_gate.set(i, sigmoid_d(igates.get(i) + hgates.get(i)));
        input_gate.set(i, sigmoid_d(igates.get(h_size + i) + hgates.get(h_size + i)));
        new_gate.set(i, tanh_d(igates.get(h_size * 2 + i) + reset_gate.get(i) * (hgates.get(h_size * 2 + i) / BIT1)));

        // h_next
        h.set(i, new_gate.get(i) + input_gate.get(i) * ((h.get(i) - new_gate.get(i)) / BIT1));
    }
}
