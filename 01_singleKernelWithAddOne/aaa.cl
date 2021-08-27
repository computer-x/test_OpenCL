__kernel void sample_test_1(__global int *src, __global int *dst)
{
    int tid = get_global_id(0);
    dst[tid] = src[tid] + 1;
}
