#ifndef __POOLING_H__
#define __POOLING_H__

class Pooling {
private:
    int c, iw, ih, s, ow, oh;
    int *p;
    double *x, *y, *d;

public:
    Pooling(int channels, int height, int width, int stride);
    ~Pooling();
    double* forward(double *input);
    double* backward(double *delta, double lr);
    void dump();

protected:
    void init_params(int idx=0);
    int inline ii(int c, int h, int w) { return c*ih*iw + h*iw + w; }
    int inline oi(int c, int h, int w) { return c*oh*ow + h*ow + w; }
};

#endif
