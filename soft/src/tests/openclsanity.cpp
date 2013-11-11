#include "oclbackend.h"
#include <iostream>

int main()
{
	PJWFront::ocl::OCLBackend backend;

	const char * kernels = "__kernel void vector_add_gpu (__global const float* src_a, __global const float* src_b, __global float* res, const int num)"
							"{"
							"const int idx = get_global_id(0);"
							"if (idx < num)"
							"res[idx] = src_a[idx] + src_b[idx]; } ";
/*
	backend.createProgram(kernels);
	
	float *cpu_a, *cpu_b, *cpu_res;
	
	const int N = 10;
	
	cpu_a = new float[N];
	cpu_b = new float[N];
	cpu_res = new float[N];
	
	for(int i=0; i<N; i++)
	{
		cpu_a[i] = (float) i;
		cpu_b[i] = (float) (N-i)*2;
		cpu_res[i] = 0;
	}
	
	cl_mem src_a = backend.sendData((void*)cpu_a, sizeof(float) * N);
	cl_mem src_b = backend.sendData((void*)cpu_b, sizeof(float) * N);
	cl_mem res = backend.sendData((void*)cpu_res, sizeof(float) * N);
	
	cl_kernel vector_add_gpu = backend.getNamedKernel("vector_add_gpu");
	
	backend.arg(vector_add_gpu, 0, src_a);
	backend.arg(vector_add_gpu, 1, src_b);
	backend.arg(vector_add_gpu, 2, res);
	backend.arg(vector_add_gpu, 3, N);
	
	backend.enqueueKernel(vector_add_gpu, N, N);
	
	backend.finish();
	
	float* ret = new float[N];
	
	backend.receiveData(res, ret, sizeof(float) * N);
	
	backend.finish();
	
	for(int i=0; i<N; i++)
	{
		cout << ret[i] << " ";
	}
	cout << endl;
*/	
	return 0;
}
