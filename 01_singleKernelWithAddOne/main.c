#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "OpenCL/opencl.h"

#ifdef WIN32
#define clmPROGRAMFILE ".\\src\\mathop.clgcSL"
#else
#define clmPROGRAMFILE "./src/mathop.clgcSL"
#endif
#define clmUSEBINARY 1

#define clmCHECKERROR(a, b) checkError(a, b, __FILE__ , __LINE__)

typedef struct OCL_ctx {
    cl_uint num_platforms;
    cl_platform_id *platforms;
    cl_uint num_devices;
    cl_device_id *devices;
    
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
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
        printf("cannot open %s\n", filename);
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
        printf("cannot read %s\n", filename);
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
    
    //err = clGetPlatformIDs(ctx->num_platforms, ctx->platforms, NULL);
    err = clGetPlatformIDs(ctx->num_platforms, ctx->platforms, NULL);
    if(err){
        printf("Set platform failed\n");
        return -1;
    }
    
    //这里应该是0，还是1？
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
    
    //第三个参数指的是要出初始化的设备的总数量
    err = clGetDeviceIDs(ctx->platforms[0], CL_DEVICE_TYPE_GPU, ctx->num_devices, ctx->devices, NULL);
    //err = clGetDeviceIDs(ctx->platforms[0], CL_DEVICE_TYPE_GPU, 1, ctx->devices, NULL);
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
    
    ctx->context = clCreateContext(NULL, 1, &ctx->devices[1], NULL, NULL, &err);
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

    //ctx->program = clCreateProgramWithSource(ctx->context, 1, (const char **)&sourcefile, NULL, &err);
    ctx->program = clCreateProgramWithSource(ctx->context, 1, (const char **)&sourcefile, &length, &err);
    if(err){
        printf("create program failed!\n");
        return -1;
    }
    
    err = clBuildProgram(ctx->program, 1, &ctx->devices[1], "-w", NULL, NULL);
    if(err){
        printf("build program failed!\n");
        return -1;
    }
    
    ctx->kernel = clCreateKernel(ctx->program, "sample_test_1", &err);
    if(err){
        printf("create kernel failed!\n");
        return -1;
    }
    
    free(sourcefile);
    return 0;
}



int main()
{
    size_t data_count = 16;
    
    cl_int err;
    OCL_ctx *ctx = (OCL_ctx *)malloc(sizeof(OCL_ctx));
    
    err = create_device(ctx);
    err = create_kernel(ctx);
    
    ctx->dsrc = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY, data_count * sizeof(float), NULL, NULL);
    ctx->ddst = clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY, data_count * sizeof(float), NULL, NULL);

    err = clSetKernelArg(ctx->kernel, 0, sizeof(cl_mem), &ctx->dsrc);
    err |= clSetKernelArg(ctx->kernel, 1, sizeof(cl_mem), &ctx->ddst);
    if(err){
        printf("set kernel arg failed!\n");
        return -1;
    }
    
    
    int hsrc[16];
    for(int i = 0; i < 16; ++i) {
        hsrc[i] = i;
    }
    
    err = clEnqueueWriteBuffer(ctx->queue, ctx->dsrc, CL_TRUE, 0, 16 * sizeof(int), hsrc, 0, NULL, NULL);
    
    
    size_t global[1] = { 16 };
    size_t local[1] = { 1 };
    
    err = clEnqueueNDRangeKernel(ctx->queue, ctx->kernel, 1, NULL, global, local, 0, NULL, NULL);
    
    
    int hdst[16];
    err = clEnqueueReadBuffer(ctx->queue, ctx->ddst, CL_TRUE, 0, 16 * sizeof(int), hdst, 0, NULL, NULL);
    
    
    for(int i = 0; i < 16; ++i){
        printf("%d, ", hdst[i]);
    }
    printf("\n");
    

    free(ctx);
    if(err){
        printf("error!\n");
    }
    else{
        printf("finish!\n");
    }
    return err;
}
