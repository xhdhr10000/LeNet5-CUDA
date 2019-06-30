#ifndef __CONV_H__
#define __CONV_H__

class Conv {
private:
    int ic, oc, ih, iw, oh, ow, k, s, p;
    float *w, *b, *x, *y, *d, *dw, *db;

public:
    Conv(int input_channels, int input_height, int input_width, int output_channels, int kernel, int stride, int padding);
    ~Conv();
    float* forward(float *input);
    float* backward(float *delta, float lr);
    void dump();

protected:
    void init_params();
    int inline ii(int c, int h, int w) { return c*ih*iw + h*iw + w; }
    int inline oi(int c, int h, int w) { return c*oh*ow + h*ow + w; }
    int inline wi(int o, int i, int h, int w) { return o*ic*k*k + i*k*k + h*k + w; }
};

#endif