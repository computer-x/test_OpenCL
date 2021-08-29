__kernel void add_bias(__global int *src, __global int *dst, int bias)
{
    //int gnmx = get_global_size(0);
    //int gnmy = get_global_size(1);
    //printf("gnmx = %d, gnmy = %d\n", gnmx, gnmy);
    //上述测试证明了，第0维确实对应着coller中的global(0)
    
    int tidx = get_global_id(0);
    int tidy = get_global_id(1);
    dst[tidy * 8 + tidx] = src[tidy * 8 + tidx] + bias;

    //但是第0维到底是raw方向，还是col方向？因为这两种计算方法都对
    //只能通过运行时间来反映哪一种没有造成线程束分化
    //目前根据资料显示，应该是上面没有被注释的那种算法合理一些
    //dst[tidx * 4 + tidy] = src[tidx * 4 + tidy] + bias;
}

__kernel void mul_times(__global int *src, __global int *dst, int times)
{
    int tidx = get_global_id(0);
    int tidy = get_global_id(1);
    dst[tidy * 8 + tidx] = src[tidy * 8 + tidx] * times;
    
    //dst[tidx * 4 + tidy] = src[tidx * 4 + tidy] * times;
}
