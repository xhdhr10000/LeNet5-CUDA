#ifndef __DENSE_H__
#define __DENSE_H__

class Dense {
private:
    int ic, oc;
    float *w, *b, *x, *y, *d, *dw, *db;

public:
    Dense(int input_channels, int output_channels);
    ~Dense();
    float* forward(float *input);
    float* backward(float *delta, float lr);
    void dump();

protected:
    void init_params();
};

#endif