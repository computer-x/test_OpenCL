__kernel void add_bias(__global int *src, __global int *dst, int bias)
{
    int tid = get_global_id(0);
    dst[tid] = src[tid] + bias;
}

__kernel void mul_times(__global int *src, __global int *dst, int times)
{
    int tid = get_global_id(0);
    dst[tid] = src[tid] * times;
}
