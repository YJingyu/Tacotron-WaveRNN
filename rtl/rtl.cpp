#include <chrono>

#include "layer.hpp"

using namespace std;

Array <int, SAMPLE_SIZE>            out;
Array <int, 1 + MELS_DIM + AUX_DIM> I;
Array <int, HIDDEN_SIZE + AUX_DIM>  inp;
Array <int, HIDDEN_SIZE>            x;
Array <int, HIDDEN_SIZE>            h1;
Array <int, HIDDEN_SIZE>            h2;
Array <long, HIDDEN_SIZE>           p;

int sample = 0;

void debug() {
    out.print("out");
    I.print("I");
    inp.print("inp");
    x.print("x");
    p.print("p");
    h1.print("h1");
    h2.print("h2");
    igates.print("igates");
    hgates.print("hgates");
    reset_gate.print("reset_gate");
    input_gate.print("input_gate");
    new_gate.print("new_gate");
}

int main() {
    printf("***** Start WaveRNN inference *****\n");

    // load inputs
    printf("Loading inputs from file...\n");
    loadfile();

    // inference loop
    auto start = chrono::system_clock::now();
    printf("Enter inference loop!!!\n");
    for (int i = 0; i < SAMPLE_SIZE; ++i) {
        // I
        concat(I, sample, mels[i], aux_0[i]);
        linear_I(x, I);

        // rnn1
        rnn_1(x, h1);
        add(x, h1);

        // rnn2
        concat_2(inp, x, aux_1[i]);
        rnn_2(inp, h2);
        add(x, h2);

        // fc1
        concat_2(inp, x, aux_2[i]);
        linear_fc1(x, inp);
        relu(x);

        // fc2
        concat_2(inp, x, aux_3[i]);
        linear_fc2(x, inp);
        relu(x);

        // fc3
        linear_fc3(p, x);
        softmax(p);

        // categorize
        sample = (choice(p) * BIT2 * 2) / (HIDDEN_SIZE - 1) - BIT2;
        out.set(i, sample);

        // show progress
        if ((i + 1) % (SAMPLE_SIZE / 10) == 0) {
            int  progress = (i + 1) / (SAMPLE_SIZE / 100);
            auto end      = chrono::system_clock::now();
            auto sec      = chrono::duration <double>(end - start).count();
            printf("|%7.2lf s||%3d %%|", sec, progress);
            for (int j = 0; j < (progress / 10); ++j) {
                printf("##");
            }
            printf("\n");
        }
    }

    // save outputs
    printf("Saving outputs to file...\n");
    savefile(out);

    // debug
    debug();

    printf("***** Finish WaveRNN inference *****\n");

    return 0;
}
