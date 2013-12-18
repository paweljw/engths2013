#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable

void atomic_add_local(volatile global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
 
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
 
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

inline int CmGet(
		const unsigned int row,
		const unsigned int col,
		__global unsigned int *N){
	
	return ((row * *N) + col);
}
inline unsigned int ReduceRows(
		const unsigned int original,
		const unsigned int offender,
		const unsigned int function,
		__global float* dataMatrix,
		__global float* dataRhs,
		__global unsigned int *N)
	{	
	float multiplier = dataMatrix[original * *N + function] / dataMatrix[offender * *N + function];
	multiplier *= -1;
	
	unsigned int flop = 2;
	
	for(unsigned int i = function; i < *N; i++)
	{
			atomic_add_local(&dataMatrix[original * *N + i], (dataMatrix[offender * *N + i] * multiplier));
			flop += 2;
	}
				
	dataRhs[original] += dataRhs[offender] * multiplier;
	flop += 2;
    
	return flop;
}
inline float ___abs(float val) 
{ 
	if(val < 0) 
		return val*-1.0f; 
	return val; 
}

inline unsigned int RowFunction(
		unsigned int row,
		__global float* dataMatrix,
		__global unsigned int *N)
		{
		
		unsigned int beginAt = row * *N;
		
		for(unsigned int ix = 0; ix < *N; ix++)
		{
			if(___abs(dataMatrix[beginAt+ix]) <= --TAG_NUMERICAL_ERROR--) 
				dataMatrix[beginAt+ix] = 0;
			if(dataMatrix[beginAt+ix] != 0) 
				return ix;
		}
		
	return (*N);
}

__kernel void Mangler(
	__global float* dataMatrix,
	__global float* dataRhs, 
	__global int* map, 
	__global unsigned int* N,
	__global unsigned int *flop) 
{
	__local int localMap[--TAG_LOCAL_MAP_SIZE--];
	
	unsigned int local_flop = 1;
	
	int threadID = get_local_id(0);
	int blockID = get_group_id(0);
	int param_block_size = get_local_size(0);
	
	if(0 == threadID) 
	{
		for(int i=0; i<(*N); i++) 
				localMap[i] = -1;
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	int rnumber = blockID * param_block_size + threadID;
	
	if(rnumber < (*N))
	{
		while(true){
			int function = RowFunction(rnumber, dataMatrix, N);
			if(function == *N) break;
			int offender = atomic_cmpxchg(&(localMap[function]), -1, rnumber);
			if(offender != -1)
				local_flop += ReduceRows(rnumber, offender, function, dataMatrix, dataRhs, N);
			else break;
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);

		if(threadID == 0)
		{
			for(int i=0; i<(*N); i++)
			{
				map[blockID*(*N)+i] = localMap[i];
			}
		}
	}
	
	atomic_add(flop, local_flop);
}
__kernel void Resolver(
		__global float* dataMatrix,
		__global float* dataRhs,
		__global unsigned int* map,
		__global unsigned int *N,
		__global unsigned int *BLOX,
		__global unsigned int *ops,
		__global unsigned int *flop)
	{
		int threadID = get_local_id(0);
		int blockID = get_group_id(0);
		int param_block_size = get_local_size(0);
		
		int row = threadID + blockID * param_block_size;
		
		if(row < *N)
		{
			int first = -1;
			int function = -1;
			
			unsigned int lops = 0;
			unsigned int local_flop = 0;
			
			for(int block=0; block<(*BLOX); block++)
			{
				if(map[block*(*N)+row] != -1)
				{
					if(first == -1){
						first = map[block*(*N)+row];
						function = row;
						continue;
					} else {
						int thisRow = map[block*(*N)+row];
						local_flop += ReduceRows(thisRow, first, function, dataMatrix, dataRhs, N);
						lops++;
					}
				}
			}
			
			atomic_add(ops, lops);
			atomic_add(flop, local_flop);
		}
		
	}