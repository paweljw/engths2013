inline float ___abs(float val) 
{ 
	if(val < 0) 
		return val*-1.0f; 
		
	return val;
}

inline unsigned int RowFunction(unsigned int row_number, __global float* dataMatrix, __global unsigned int* N)
{
	int row_function = 0;
	
	for(unsigned int i=0; i < *N; i++)
	{
		if(isnan(dataMatrix[row_number * *N + i])) dataMatrix[row_number * *N + i] = 0;
		if(___abs(dataMatrix[row_number * *N + i]) < 0.0000001) dataMatrix[row_number * *N + i] = 0;
		if(dataMatrix[row_number * *N + i] != 0) return i;
	}
	
	return *N;
}

inline void ReduceRows(
	const unsigned int original,
	const unsigned int offender,
	const unsigned int function,
	__global float* dataMatrix,
	__global float* dataRhs,
	__global unsigned int *N)
	{
		
		float multiplier = dataMatrix[original * *N + function] / dataMatrix[offender * *N + function];
		multiplier *= -1;
		
		for(unsigned int i = function; i < *N; i++)
		{
			dataMatrix[original * *N + i] += dataMatrix[offender * *N + i];
		}
		
		dataRhs[original] += dataRhs[offender];
	}

__kernel void Mangler(
	__global float* dataMatrix, 
	__global float* dataRhs, 
	__global int* map, 
	__global unsigned int* N)
	{
		int row_number = get_local_id(0) + get_group_id(0) * get_local_size(0);
		/*
		if(row_number < *N)
		{
			while(true)
			{
				unsigned int function = RowFunction(row_number, dataMatrix, N);
				if(function == (*N)) break;
				
				int offender = atomic_cmpxchg(&(map[function]), -1, row_number);
				
				if(offender != -1)
					ReduceRows(row_number, offender, function, dataMatrix, dataRhs, N);
				else break;
			}
		}
		*/
	}