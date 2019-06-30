#ifndef __DENSE_H__
#define __DENSE_H__

class Dense {
private:
    int ic, oc;
    double *w, *b, *x, *y, *d, *dw, *db;

public:
    Dense(int input_channels, int output_channels);
    ~Dense();
    double* forward(double *input);
    double* backward(double *delta, double lr);
    void dump();

protected:
    void init_params();
};

#endif