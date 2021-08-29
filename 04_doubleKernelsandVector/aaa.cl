__kernel void add_bias(__global int8 *src, __global int8 *dst, __global int8 *bias)
{
    int tidx = get_global_id(0);
    dst[tidx] = src[tidx] + *bias;
}

__kernel void mul_times(__global int8 *src, __global int8 *dst, __global int8 *times)
{
    int tidx = get_global_id(0);
    dst[tidx] = src[tidx] * *times;
}
