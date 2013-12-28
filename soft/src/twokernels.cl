#if defined(cl_amd_fp64) || defined(cl_khr_fp64)
	#ifdef cl_amd_fp64
		#pragma OPENCL EXTENSION cl_amd_fp64 : enable
	#elifdef cl_khr_fp64
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
	#endif
#else
	#define float double
#endif

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline int CmGet(
	const unsigned int i,
	const unsigned int j,
	__global unsigned int* rowsPtr,
	__global unsigned int* fnIds,
	__global unsigned int *N)
	{
		if(i >= *N || j >= *N) return -1;
        	unsigned int offset = rowsPtr[i];
        	int colOffset = j - fnIds[i];
		if(colOffset < 0)
				return -1;
		return offset+colOffset;
    	}

inline unsigned int ReduceRows(
		const unsigned int original,
		const unsigned int offender,
		const unsigned int function,
		__global double* dataMatrix,
		__global unsigned int* rowsPtr, 
		__global unsigned int* fnIds,
		__global double* dataRhs,
		__global unsigned int *N)
	{	
	double multiplier = dataMatrix[CmGet(original, function, rowsPtr, fnIds, N)] / dataMatrix[CmGet(offender, function, rowsPtr, fnIds, N)];
	multiplier *= -1;
	
	unsigned int flop = 2;

//	printf("Reduce %d against %d with mult of %f\n", original, offender, multiplier);
	
	for(unsigned int i = function; i < *N; i++)
	{
		if(i==function) 
			dataMatrix[CmGet(original, i, rowsPtr, fnIds, N)] = 0;
		else 
		{
			int origPos = CmGet(original, i, rowsPtr, fnIds, N);
			int offPos = CmGet(offender, i, rowsPtr, fnIds, N);

  		//        printf("-> Reduction round %d, posdata %d:%d, data %f:%f\n", i, origPos, offPos, origPos == -1 ? 0.0f : dataMatrix[origPos], offPos == -1 ? 0.0f : dataMatrix[offPos]);
			
			if(offPos == -1 || origPos == -1) 
				continue;
			
			dataMatrix[origPos] += (dataMatrix[offPos] * multiplier);
		}			
		flop += 2;
	}
				
	dataRhs[original] += dataRhs[offender] * multiplier;
	flop += 2;
    
	return flop;
	}

inline double ___abs(double val) 
{ 
	if(val < 0) 
		return val*-1.0f; 
	return val; 
}

inline unsigned int RowFunction(
		unsigned int row,
		__global double* dataMatrix,
		__global unsigned int* rowsPtr, 
		__global unsigned int* fnIds,
		__global unsigned int *N)
		{
		if(!(row < *N)) return *N;
		
		unsigned int beginAt = fnIds[row];
		for(unsigned int ix = beginAt; ix < *N; ix++)
		{
			if(CmGet(row, ix, rowsPtr, fnIds, N) == -1) continue;

			if(___abs(dataMatrix[CmGet(row, ix, rowsPtr, fnIds, N)]) <= --TAG_NUMERICAL_ERROR--) 
				dataMatrix[CmGet(row, ix, rowsPtr, fnIds, N)] = 0;

			if(isnan(dataMatrix[CmGet(row, ix, rowsPtr, fnIds, N)]))
				dataMatrix[CmGet(row, ix, rowsPtr, fnIds, N)] = 0;

			if(dataMatrix[CmGet(row, ix, rowsPtr, fnIds, N)] != 0) 
				return ix;
		}
		
	return (*N);
}

__kernel void Mangler(
	__global double* dataMatrix,
	__global unsigned int* rowsPtr, 
	__global unsigned int* fnIds,
	__global double* dataRhs, 
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
	
	int rnumber = blockID * param_block_size + threadID;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(rnumber < (*N))
	{
		while(true){
			int function = RowFunction(rnumber, dataMatrix, rowsPtr, fnIds, N);
			if(function == *N) break;
			int offender = atomic_cmpxchg(&(localMap[function]), -1, rnumber);
		
		//	printf("row %d function %d offender %d\n", rnumber, function, offender);	
			if(offender != -1)
				local_flop += ReduceRows(rnumber, offender, function, dataMatrix, rowsPtr, fnIds, dataRhs, N);
			
			else break;
		}
	}
		
	barrier(CLK_LOCAL_MEM_FENCE);
		
	if(threadID == 0)
	{
		for(int i=0; i<(*N); i++)
		{
			map[blockID*(*N)+i] = localMap[i];
		}
	}
	
	atomic_add(flop, local_flop);
}
__kernel void Resolver(
		__global double* dataMatrix,
		__global unsigned int* rowsPtr, 
		__global unsigned int* fnIds,
		__global double* dataRhs,
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
						local_flop += ReduceRows(thisRow, first, function, dataMatrix, rowsPtr, fnIds, dataRhs, N);
						lops++;
					}
				}
			}
			
			atomic_add(ops, lops);
			atomic_add(flop, local_flop);
		}
		
	}
