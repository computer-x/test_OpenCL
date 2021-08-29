/*
        chenxu   2021.8.29
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "OpenCL/opencl.h"


typedef struct OCL_ctx {
    cl_uint num_platforms;
    cl_platform_id *platforms;
    cl_uint num_devices;
    cl_device_id *devices;
    
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel *kernel;
    cl_uint num_kernels;
    cl_mem dsrc, ddst;
} OCL_ctx;

char *read_source_file(const char *filename, size_t *length)
{
    FILE *fp = NULL;
    size_t sourceLength;
    //int sourceLength;
    char *sourceString;
    unsigned long ret;
    fp = fopen(filename, "rb");
    if(fp == NULL){
        printf("cannot open %s!\n", filename);
        return NULL;
    }
    
    fseek(fp, 0, SEEK_END);
    sourceLength = ftell(fp);
    if(length != NULL){
        *length = sourceLength;
    }
    
    fseek(fp, 0, SEEK_SET);
    sourceString = (char *)malloc(sourceLength + 1);
    sourceString[0] = '\0';
    ret = fread(sourceString, sourceLength, sizeof(char), fp);
    if(ret == 0){
        printf("cannot read %s!\n", filename);
        return NULL;
    }
    
    fclose(fp);
    
    sourceString[sourceLength] = '\0';
    
    return sourceString;
}

int create_device(OCL_ctx *ctx)
{
    cl_int err;
    
    err = clGetPlatformIDs(0, NULL, &ctx->num_platforms);
    if(err){
        printf("Get Platform number failed!\n");
        return -1;
    }
    printf("num_platform: %d\n", ctx->num_platforms);
    
    ctx->platforms = (cl_platform_id *) malloc(ctx->num_platforms * sizeof(cl_platform_id));
    if(!ctx->platforms){
        printf("platform malloc failed!\n");
        return -1;
    }
    
    err = clGetPlatformIDs(ctx->num_platforms, ctx->platforms, NULL);
    if(err){
        printf("Set platform failed!\n");
        return -1;
    }
    
    err = clGetDeviceIDs(ctx->platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &ctx->num_devices);
    if(err){
        printf("Get devices number failed!\n");
        return -1;
    }
    printf("num_device（GPU）: %d\n", ctx->num_devices);
    
    ctx->devices = (cl_device_id *)malloc(ctx->num_devices * sizeof(cl_device_id));
    if(!ctx->devices){
        printf("device malloc failed!\n");
        return -1;
    }
    
    //标准写法，第三个参数指的是要初始化的设备的总数量
    err = clGetDeviceIDs(ctx->platforms[0], CL_DEVICE_TYPE_GPU, ctx->num_devices, ctx->devices, NULL);
    //按理说我用哪个设备就初始化那个就行了，但是无论我初始化&ctx->devices[0]还是初始化&ctx->devices[1]，最后查询出来都是Intel UDH Graphics，原因尚不明
    //err = clGetDeviceIDs(ctx->platforms[0], CL_DEVICE_TYPE_GPU, 1, &ctx->devices[1], NULL);
    
    if(err){
        printf("get device id failed!\n");
        return -1;
    }

    char device_name[100];
    err = clGetDeviceInfo(ctx->devices[0], CL_DEVICE_NAME, 100, device_name, NULL);
    if(err){
        printf("get device info failed!\n");
        return -1;
    }
    printf("device_name[0]: %s\n", device_name);
    
    err = clGetDeviceInfo(ctx->devices[1], CL_DEVICE_NAME, 100, device_name, NULL);
    if(err){
        printf("get device info failed!\n");
        return -1;
    }
    printf("device_name[1]: %s\n", device_name);
    
    return 0;
}

int create_kernel(OCL_ctx *ctx)
{
    cl_int err;
    
    ctx->context = clCreateContext(NULL, 1, &ctx->devices[1], NULL, NULL, &err); //创建上下文时选的设备 需要与 创建上下文时所选的设备 一致
    if(err){
        printf("create context failed!\n");
        return -1;
    }
    
    cl_device_id test; //因为创建上下文的时候，已经选定设备了，所以这里只能有一个，如果创建多个，则后面的全是空。
    err = clGetContextInfo(ctx->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &test, NULL);
    char device_name[100];
    err = clGetDeviceInfo(test, CL_DEVICE_NAME, 100, device_name, NULL);
    if(err){
        printf("get device info failed!\n");
        return -1;
    }
    printf("device used by context: %s\n", device_name);
    
    ctx->queue = clCreateCommandQueue(ctx->context, ctx->devices[1], 0, &err); //创建命令队列时选的设备 需要与 创建上下文时所选的设备 一致
    if(err){
        printf("create command queue failed!\n");
        return -1;
    }
    
    size_t length;
    char *sourcefile = read_source_file("/Users/chenxu/project_xcode/mult/mult/aaa.cl", &length);
    if(length == 0){
        printf("read source file failed!\n");
        return -1;
    }
    //printf("kernel(length:%d):\n%s\n", length, sourcefile);

    ctx->program = clCreateProgramWithSource(ctx->context, 1, (const char **)&sourcefile, NULL, &err); //这里要不要这个长度信息无所谓
    //ctx->program = clCreateProgramWithSource(ctx->context, 1, (const char **)&sourcefile, &length, &err);
    if(err){
        printf("create program failed!\n");
        return -1;
    }
    
    err = clBuildProgram(ctx->program, 1, &ctx->devices[1], "-w", NULL, NULL);
    if(err){
        printf("build program failed!\n");
        return -1;
    }
    
    err = clCreateKernelsInProgram(ctx->program, 0, NULL, &ctx->num_kernels);
    if(err){
        printf("get kernel number failed!\n");
        return -1;
    }
    
    printf("kernels number: %d\n", ctx->num_kernels);
    ctx->kernel = (cl_kernel *)malloc(ctx->num_kernels * sizeof(cl_kernel));
    
//#define ONEBYONE
#ifdef ONEBYONE
    printf("=== Create Kernel One by One ===\n"); //这种方式比较麻烦，但是比较直观、稳定
    ctx->kernel[0] = clCreateKernel(ctx->program, "add_bias", &err);
    if(err){
        printf("create kernel 'add_bias' failed!\n");
        return -1;
    }
    ctx->kernel[1] = clCreateKernel(ctx->program, "mul_times", &err);
    if(err){
        printf("create kernel 'mul_times' failed!\n");
        return -1;
    }
#else
    printf("=== Create Kernels in Program ===\n"); //这种方法可以一次性自动创建所有kernels，
                                                   //但是内核列表中的名称顺序并不是由文件中的书写顺序决定，而是依赖于实现
                                                   //因此需要查询每个内核对象名称来确认，避免出错
    err = clCreateKernelsInProgram(ctx->program, ctx->num_kernels, ctx->kernel, NULL);
    if(err){
        printf("create kernels failed!\n");
        return -1;
    }
    char kernel_name[100];
    for(int i = 0; i < ctx->num_kernels; ++i)
    {
        err = clGetKernelInfo(ctx->kernel[i], CL_KERNEL_FUNCTION_NAME, 100, kernel_name, &length);
        if(err){
            printf("get kernel[%d] name info failed!\n", i);
            return -1;
        }
        printf("kernel[%d] name: %s (length=%d)\n", i, kernel_name, length);
    }
#endif
    
    free(sourcefile);
    return 0;
}

int ctx_free(OCL_ctx *ctx)
{
    cl_int err = 0;
    err = clReleaseMemObject(ctx->ddst);
    err |= clReleaseMemObject(ctx->dsrc);
    for(int i = 0; i < ctx->num_kernels; ++i){
        err |= clReleaseKernel(ctx->kernel[i]);
    }
    err |= clReleaseProgram(ctx->program);
    err |= clReleaseCommandQueue(ctx->queue);
    err |= clReleaseContext(ctx->context);
    
    //那么platform和device释放吗？怎么释放？
    
    free(ctx->kernel);
    free(ctx);
    return err;
}

int main()
{
    size_t data_count = 8 * 8;
    
    cl_int err;
    OCL_ctx *ctx = (OCL_ctx *)malloc(sizeof(OCL_ctx));
    
    err = create_device(ctx);
    err |= create_kernel(ctx);
    if(err){
        return -1;
    }
    
    ctx->dsrc = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY, data_count * sizeof(float), NULL, NULL);
    ctx->ddst = clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY, data_count * sizeof(float), NULL, NULL);
    cl_int dtimes = 3;
    cl_int dbias = 10; //这种形式的参数值必须在clSetKernelBuffer之前确定

    err = clSetKernelArg(ctx->kernel[1], 0, sizeof(cl_mem), &ctx->dsrc);
    err |= clSetKernelArg(ctx->kernel[1], 1, sizeof(cl_mem), &ctx->ddst);
    err |= clSetKernelArg(ctx->kernel[1], 2, sizeof(cl_mem), &dtimes);
    
    err = clSetKernelArg(ctx->kernel[0], 0, sizeof(cl_mem), &ctx->ddst);
    err |= clSetKernelArg(ctx->kernel[0], 1, sizeof(cl_mem), &ctx->ddst);
    err |= clSetKernelArg(ctx->kernel[0], 2, sizeof(cl_mem), &dbias);
    
    if(err){
        printf("set kernel arg failed!\n");
        return -1;
    }
    
    int hsrc[4 * 8], hdst[4 * 8]; //内存中的值可以在clSetKernelBuffer之后确定，但必须在clEnqueueWriteBuffer之前确定
    for(int i = 0; i < 4; ++i) {
        for(int j = 0; j < 8; ++j) {
            hsrc[i * 8 + j] = i * 8 + j;
            //hsrc[i * 8 + j] = i;
        }
    }
    
    err = clEnqueueWriteBuffer(ctx->queue, ctx->dsrc, CL_TRUE, 0, 4 * 8 * sizeof(int), hsrc, 0, NULL, NULL);
    
    size_t global[2] = { 8, 4 };
    size_t local[2] = { 8, 4 };
    err = clEnqueueNDRangeKernel(ctx->queue, ctx->kernel[1], 2, NULL, global, local, 0, NULL, NULL);
    err = clEnqueueNDRangeKernel(ctx->queue, ctx->kernel[0], 2, NULL, global, local, 0, NULL, NULL);
    
    err = clEnqueueReadBuffer(ctx->queue, ctx->ddst, CL_TRUE, 0, 4 * 8 * sizeof(int), hdst, 0, NULL, NULL);
    
    printf("result:\n");
    for(int i = 0; i < 4; ++i) {
        for(int j = 0; j < 8; ++j) {
            printf("%3d, ", hdst[i * 8 + j]);
        }
        printf("\n");
    }
    
    err = ctx_free(ctx);
    if(err){
        printf("error!\n");
    }
    else{
        printf("finish!\n");
    }
    return err;
}
