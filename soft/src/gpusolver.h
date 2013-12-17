#pragma once

#include "util.h"
#include "oclbackend.h"
#include "shortmatrix.h"
#include "oclvector.h"
#include <cfloat>

namespace PJWFront
{
	// TODO: Are all these pragmas seriously necessary?!
	
	static const char * __pjws__kernels = "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable"
	"#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable"
	"#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable"
	"#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable"
	"#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable\n"
	"#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable\n"
	"inline int CmGet(const unsigned int i,const unsigned int j,__global unsigned int* rowsPtr,__global unsigned int* fnIds,__global unsigned int *N){\n"
	"	if(i >= *N || j >= *N) return -1;"
	"	unsigned int offset = rowsPtr[i];\n"
	"	int colOffset = j - fnIds[i];\n"
	"	if(colOffset < 0)\n"
	"		return -1;\n"
	"	return offset+colOffset;\n"
	"}\n"
	"inline unsigned int ReduceRows(const unsigned int original,const unsigned int offender,const unsigned int begin,__global unsigned int* rowsPtr,__global unsigned int* fnIds,__global float* dataMatrix,__global float* dataRhs,__global unsigned int *N){\n"
	"	float multiplier = dataMatrix[CmGet(original, begin, rowsPtr, fnIds, N)] / dataMatrix[CmGet(offender, begin, rowsPtr, fnIds, N)];\n"
	"	multiplier *= -1;\n"
	"	unsigned int flop = 2;\n"	
	"	for(int i=begin; i<(*N);i++)\n"
	"	{\n"
	"		if(i==begin)\n"
	"		{ \n"
	"			int origpos = CmGet(original, i, rowsPtr, fnIds, N); \n"
	"			dataMatrix[origpos] = 0; \n"
	"			continue; \n"
	"		}\n"
	"		int origpos = CmGet(original, i, rowsPtr, fnIds, N);\n"
	"		int ofpos = CmGet(offender, i, rowsPtr, fnIds, N);\n"
	"		flop += 1;"
	"		if(ofpos == -1 || origpos == -1)\n"
	"			continue;\n"
	"		if(isnan(dataMatrix[ofpos])) dataMatrix[ofpos] = 0;"
	"		if(isnan(dataMatrix[origpos])) dataMatrix[ofpos] = 0;"
	"		dataMatrix[origpos] += dataMatrix[ofpos] * multiplier;\n"
	"	}\n"
	"	dataRhs[original] += dataRhs[offender] * multiplier;\n"
	"	flop += 1;\n"
	"   return flop;\n"
	"}\n"
	"inline float ___abs(float val) { if(val < 0) return val*-1.0f; return val; }"
	"inline unsigned int RowFunction(unsigned int row,__global float* dataMatrix,__global unsigned int* rowsPtr,__global unsigned int* fnIds,__global unsigned int *N){\n"
	"	if(row >= *N) return -1;"
	"	unsigned int beginAt = rowsPtr[row];\n"
	"	unsigned int searchUntil = (*N) - fnIds[row];\n"
	"	for(unsigned int ix = 0; ix < searchUntil; ix++){\n"
	"		if(___abs(dataMatrix[beginAt+ix]) <= --TAG_NUMERICAL_ERROR--) dataMatrix[beginAt+ix] = 0;"
	"		if(isnan(dataMatrix[beginAt+ix])) dataMatrix[beginAt+ix] = 0;"
	"		if(dataMatrix[beginAt+ix] != 0) return fnIds[row]+ix;}\n"
	"	return (*N);"
	"};\n"
	"__kernel void Mangler(__global float* dataMatrix, __global unsigned int* rowsPtr, __global unsigned int* fnIds, __global float* dataRhs, __global int* map, __global unsigned int* N, __global const unsigned int *param_block_size, __global unsigned int *flop) \n"
	"{\n"
	"	__local int localMap[--TAG_LOCAL_MAP_SIZE--];\n"
	"	atomic_add(flop, 1);\n"
	"	int threadID = get_local_id(0);\n"
	"	int blockID = get_group_id(0);\n"
	"	if(0 == threadID) \n"
	"	{\n"
	"		for(int i=0; i<(*N); i++) \n"
	"				localMap[i] = -1;\n"
	"	}\n"
	"	barrier(CLK_LOCAL_MEM_FENCE);\n"
	"	int rnumber = blockID * (*param_block_size) + threadID;\n"
	"	if(rnumber < (*N))"
	"		while(true){"
	"			int function = 0;\n"
	"			function = RowFunction(rnumber, dataMatrix, rowsPtr, fnIds, N);\n"
	"			if(function < 0 || function >= (*N)) break;\n"
	"			int offender = atomic_cmpxchg(&(localMap[function]), -1, rnumber);\n"
	"			if(offender != -1)\n"
	"				atomic_add(flop, ReduceRows(rnumber, offender, function, rowsPtr, fnIds, dataMatrix, dataRhs, N));\n"
	"			else break;\n"
	"		}\n"
	"	barrier(CLK_LOCAL_MEM_FENCE);\n"
	"	if(threadID == 0)\n"
	"	{\n"
	"		for(int i=0; i<(*N); i++)\n"
	"		{\n"
	"			map[blockID*(*N)+i] = localMap[i];\n"
	"		}\n"
	"	}\n"
	"}\n"
	"__kernel void Resolver(__global float* dataMatrix,__global unsigned int* rowsPtr,__global unsigned int* fnIds,__global float* dataRhs,__global unsigned int* map,__global unsigned int *N,__global unsigned int *BLOX,__global unsigned int *ops, __global unsigned int *flop){\n"
	"	int row = get_local_id(0);\n"
	"	if(row < (*N)){"
	"	int first = -1;\n"
	"	int function = -1;\n"
	"	int lops = 0;\n"
	"	for(int i=0; i<(*BLOX); i++)\n"
	"	{\n"
	"		if(map[i*(*N)+row] != -1)\n"
	"		{\n"
	"			if(first == -1){\n"
	"				first = map[i*(*N)+row];\n"
	"				function = RowFunction(row, dataMatrix, rowsPtr, fnIds, N);\n"
	"				continue;\n"
	"			} else {\n"
	"				int thisRow = map[i*(*N)+row];\n"
	"				atomic_add(flop, ReduceRows(thisRow, first, function, rowsPtr, fnIds, dataMatrix, dataRhs, N));\n"
	"				lops++;\n"
	"			}\n"
	"		}\n"
	"	}\n"
	"	atomic_add(ops, lops);}"
	"}\n";

	/// @brief Main class for the GPU version of the frontal solver
	/// @tparam ScalarType The scalar type that will be used throughout the CPU and GPU code to store scalar values. 
	/// Currently only float is supported as double support is widely available on NVidia chips only
	/// @author Pawel J. Wal
	template <typename ScalarType>
	class GPUFrontal
	{
	private:
		/// Number of processing blocks (CUDA: workgroups, OCL: workitems).
		uint BLOCK_NUM;
		
		/// CPU-to-GPU data matrix structure
		util::short_matrix<ScalarType>	gpu_matrix;
		/// CPU-to-GPU RHS storage structure
		util::ocl_vector<ScalarType>	gpu_rhs;
		/// CPU-to-GPU mapping storage structure
		util::ocl_vector<int>	gpu_map;

		/// CPU-only matrix storage
		std::vector< std::map< unsigned int, ScalarType> > cpu_matrix;

		/// Size of an NxN matrix
		uint N;

		/// Numerical error storage
		ScalarType NUM_ERR;

		/// A backend helper pointer storage
		ocl::OCLBackend* backend;
		
		/// A simple helper function for backsubstitution phase
		/// @param sm A synchronized map pulled from the GPU after completing solution cycle
		/// @param fn The function for which we need to find the row
		/// @return The row in the matrix and RHS where the function is stored
		int functionAt(unsigned int* sm, unsigned int fn)
		{
			for(uint i=0; i<N; i++)
				if(sm[i] == fn) return i;
				
			return -1; // This means trouble and usually causes an exception to be thrown
		}

	public:
		/// FMADs number external storage for automated testing
		double fmads;
		/// Total time spent in OCL
		double ocltime;

		/// Global Work Size
		uint GWS;
		/// Local Work Size (workgroup/workitem size)
		uint LWS;

		/// Solution vector storage
		std::vector<ScalarType> solution;
		
		/// compat
		GPUFrontal()
		{
		}
		
		/// Main constructor
		/// @param size The size of the matrix
		/// @param _LWS Local work size, autocalculates unless nonzero provided
		/// @param _GWS Global work size, autocalculates unless nonzero provided
		/// @param _NE Allowable numerical error (basically tells GPU abs(number) < numerical_error -> number = 0), set to a safe value unless provided by hand
		/// @remark If an unreasonable value is provided for _LWS parameter, the constructor will override user's decision and try to salvage the situation.
		GPUFrontal(int size, uint _LWS = 192, uint _GWS = 0, ScalarType _NE = 0.0000001)
		{
			// Initialize OpenCL backend
			backend = new ocl::OCLBackend();

			// Remember matrix size
			N = size;

			// Remember numerical error
			NUM_ERR = _NE;
	
			// If LWS is not provided, it is calculated, somewhat forcefully
			// An LWS larger than 32 will give poor performance, since that's usually the max workgroup size on NVidia cards
			// NOTE: Above needs further verification - conflicting data (32 max size WG vs 192 microcores
			if(_LWS == 0)
				for(_LWS = N-1; _LWS > 0; _LWS--)
					if(!(N % _LWS)) break;
			
			// If N is a prime number, then we're in a bit of a bind since _LWS will be calculated as 1
			// Having a local work size of 1 is about as wasteful as it can be on most OpenCL devices
			// We'll just use one big block containing all the threads
			// This is also wasteful, but saves the GPU from switching contexts on processors so much
			// It will, however, give every multiprocessor a hard time, since
			// 		a) it will have to slice it's own work into pieces smaller than 32
			//		b) OpenCL probably can't split such a workgroup for multiple SMPs
			//		c) the kernels used here have a lot of divergence points, which is a problem when they're executed in large WGs
			if(_LWS < 2)
				_LWS = _GWS == 0 ? N : _GWS;
				
			LWS = _LWS;

			// We do need GWS >= N, or the method won't work
			// If someone sets it explicitly to be >N, figure they know what they doing
			// Otherwise, calculate next best thing (first number divisible by LWS larger than N)
			while(_GWS < N)		
				_GWS += LWS;

			GWS = _GWS;

			// Block number is automatically calculated
			BLOCK_NUM = GWS/LWS;

#if defined(__SOLVERDEBUG) || defined(__SOLVERTIMING)
			cout << "BN: " << BLOCK_NUM << ", GWS: " << GWS << ", LWS: " << LWS << endl;
#endif

			// Initialize all the storages
			gpu_matrix = util::short_matrix<ScalarType>(N, backend);
			cpu_matrix = std::vector< std::map< unsigned int, float> > (N*N);
			gpu_rhs = util::ocl_vector<ScalarType>(N, backend);
			gpu_map = util::ocl_vector<int>(BLOCK_NUM * N, backend);
			solution = std::vector<ScalarType> (N);
			
			// Precompile the kernels
			CompileKernels();
		}

		/// Setter for matrix values
		/// @param row Matrix row
		/// @param col Matrix column
		/// @param val The value to store at given coordinates
		void set(int row, int col, ScalarType val)
		{
			cpu_matrix[row][col] = val;
		}

		/// Setter for RHS values
		/// @param row RHS row
		/// @param val The value to be stored at given row
		void setRHS(int row, ScalarType val)
		{
			gpu_rhs.put(row, val);
		}

		/// Main solution function. Call ONLY after setting all matrix and RHS values as this is an in-place solution. 
		/// It will change or destroy data in gpu_matrix structure and desync it with cpu_matrix structure.
		/// @throws UnsolvableException
		/// @todo change the way that cpu_ops is synchronized between revolutions; current wastes a couble of bits every revolution
		void solve()
		{
			// Applying matrix preconditioners, if any
			Reordering();

			#pragma region OCL memory setup
			// Copy data from CPU matrix to OCL-assisted storage (data NOT YET sent here)
			gpu_matrix = cpu_matrix;

			// Set up the two kernels into objects
			cl_kernel Mangler = backend->getNamedKernel("Mangler");
			cl_kernel Resolver = backend->getNamedKernel("Resolver");
						
			// Generate all the required data handles for kernel argument
			// Note: PCI-e data transfer happening below
			cl_mem gpu_rhs_handle = gpu_rhs.ocl_handle();			
			cl_mem gpu_map_handle = gpu_map.ocl_handle();

			cl_mem gpu_N_handle = backend->sendData(&N, sizeof(uint));
			
			// TAG: B#1
			uint localN = LWS;
			cl_mem gpu_blocksize_handle = backend->sendData(&localN, sizeof(uint));
			
			uint lbn = BLOCK_NUM;
			cl_mem gpu_blocknum_handle = backend->sendData(&lbn, sizeof(uint));

			cl_mem gpu_mx_h1 = gpu_matrix.ocl_flat_handle();
			cl_mem gpu_mx_h2 = gpu_matrix.ocl_offsets_handle();
			cl_mem gpu_mx_h3 = gpu_matrix.ocl_rowfns_handle();

			unsigned int *cpu_flops = new unsigned int;
			*cpu_flops = 0;

			cl_mem gpu_flops = backend->sendData((void*)cpu_flops, sizeof(unsigned int));

			// Note: PCI-e data transfer happening above
			
			/*
			0. __global float* dataMatrix,						// Flat-format data matrix as generated by short_matrix
			1. __global unsigned int* rowsPtr,					// Row offsets for data matrix as generated by short_matrix
			2. __global unsigned int* fnIds,					// Column offset calculation data as generated by short_matrix
			3. __global float* dataRhs,							// RHS vector, typical format
			4. __global unsigned int* map,						// Global map, flat format w/o compression
			5. __global const unsigned int N,					// Matrix size (for cases where GWS > N, limiting for loops)
			6. __global const unsigned int param_block_size		// Size of a workitem (for certain map-related duties)
			*/

			// Set up parameters for first kernel; refer to above comment for expected values
			backend->arg(Mangler, 0,  gpu_mx_h1);			
			backend->arg(Mangler, 1,  gpu_mx_h2);
			backend->arg(Mangler, 2,  gpu_mx_h3);
			backend->arg(Mangler, 3,  gpu_rhs_handle);
			backend->arg(Mangler, 4,  gpu_map_handle);
			backend->arg(Mangler, 5,  gpu_N_handle);
			backend->arg(Mangler, 6,  gpu_blocksize_handle);
			backend->arg(Mangler, 7, gpu_flops);

			/*
			0. __global float* dataMatrix,						// Flat-format data matrix as generated by short_matrix
			1. __global unsigned int* rowsPtr,					// Row offsets for data matrix as generated by short_matrix
			2. __global unsigned int* fnIds,					// Column offset calculation data as generated by short_matrix
			3. __global float* dataRhs,							// RHS vector, typical format
			4. __global unsigned int* map,						// Global map, flat format w/o compression
			5. __global const unsigned int N,					// Matrix size (for cases where GWS > N, limiting for loops)
			6. __global unsigned int *BLOX,						// Number of processing blocks
			7. __global unsigned int *ops)						// Pass-by-reference; this parameter gets pulled back to check whether another rotation is needed
			*/
	
			// Set up parameters for second kernel; refer to above comment for expected values
			// Note: with my current backend and how it does cl_mem handles, sending the 7th parameter here is counterproductive
			// A function which will send new data to an existing handle in a blocking way is needed
			backend->arg(Resolver, 0, gpu_mx_h1);
			backend->arg(Resolver, 1, gpu_mx_h2);
			backend->arg(Resolver, 2, gpu_mx_h3);
			backend->arg(Resolver, 3, gpu_rhs_handle);
			backend->arg(Resolver, 4, gpu_map_handle);
			backend->arg(Resolver, 5, gpu_N_handle);
			backend->arg(Resolver, 6, gpu_blocknum_handle);
			backend->arg(Resolver, 8, gpu_flops);

			// This will store 7th parameter CPU-side
			unsigned int *cpu_ops = new unsigned int;
			
#ifdef __SOLVERDEBUG
			gpu_matrix.printMatrix();
#endif

			ocltime = 0.0f;
			#pragma endregion

			#pragma region Solution loop
			
			uint last_cpu_ops = 0;
			uint same_for = 0;
			
			do
			{
				*cpu_ops = 0;

				// Send up zeroed out ops number
				cl_mem gpu_ops = backend->sendData(cpu_ops, sizeof(unsigned int));
#if defined(__SOLVERTIMING) || defined(__SOLVERTIMING_SILENT)
				// Enqueue first kernel and let it finish
				// Time first kernel
				cl_event mangler_event = backend->enqueueEventKernel(Mangler, LWS, GWS);
				ocltime += backend->timedFinish(mangler_event);
				
				// Set last parameter to current handle for second kernel, enqueue and let it finish
				backend->arg(Resolver, 7, gpu_ops);
				
				// Time second kernel
				cl_event resolver_event = backend->enqueueEventKernel(Resolver, N, N);
				ocltime += backend->timedFinish(resolver_event);				
#else
				backend->enqueueKernel(Mangler, LWS, GWS);
				backend->finish("mangler");

				// Set last parameter to current handle for second kernel, enqueue and let it finish
				backend->arg(Resolver, 7, gpu_ops);
				backend->enqueueKernel(Resolver, N, N);
				backend->finish("resolver");
#endif


#ifdef __SOLVERDEBUG
				gpu_matrix.pull_data(gpu_mx_h1);
				gpu_matrix.printMatrix();
				
				cout << "---" << endl;
				
				gpu_map.pull_data(gpu_map_handle);
				for(uint i=0; i<N; i++) // row
				{
					for(uint j=0; j<BLOCK_NUM; j++)
						cout << gpu_map.v.at(j*N+i) << " ";
						
					cout << endl;
				}		
				
				cout << "RHS: ";
				gpu_rhs.pull_data(gpu_rhs_handle);
				for(uint i=0; i<N; i++) // row
				{
					cout << gpu_rhs.v.at(i) << " ";
				}
				cout << endl;
				cout << "[b] Cpu ops is " << *cpu_ops << endl;
#endif					

				// Download operations data
				backend->receiveData(gpu_ops, cpu_ops, sizeof(unsigned int));
				
#ifdef __SOLVERDEBUG								
				cout << "[a] Cpu ops is " << *cpu_ops << endl;
#endif

				cout << *cpu_ops << endl;

				if(*cpu_ops == last_cpu_ops)
					same_for++;
				else
				{
					last_cpu_ops = *cpu_ops;
					same_for = 0;
				}
				
				if(same_for == 10)
					throw nanex;

			// Loop ends if no operations were needed
			} while(*cpu_ops > 0);
			
			#pragma endregion

#ifdef __SOLVERTIMING
			backend->receiveData(gpu_flops, cpu_flops, sizeof(unsigned int));
			fmads = *cpu_flops / ocltime;
			printf("Time in kernels: %0.9f s\n", ocltime);
			printf("Calculated FMAD: %u\n", *cpu_flops);
			printf("FLOPS: %0.3f\n", fmads);
#endif
			
			#pragma region Syncing the memory with the GPU
			
			// Pull all the data from the GPU
			// Note: PCI-e transfer below
			gpu_matrix.pull_data(gpu_mx_h1);
			gpu_rhs.pull_data(gpu_rhs_handle);
			gpu_map.pull_data(gpu_map_handle);
			// Note: PCI-e transfer above
			
#ifdef __SOLVERDEBUG
			gpu_matrix.printMatrix();
			cout << "RHS: ";
			gpu_rhs.pull_data(gpu_rhs_handle);
			for(uint i=0; i<N; i++) // row
			{
				cout << gpu_rhs.v.at(i) << " ";
			}
			cout << endl;
#endif

			// Local storage for the synchronized (from multiple block maps) map
			unsigned int *synchronized_map = new unsigned int[N];

			for(uint row=0; row<N; row++)
				for(uint block=0; block<BLOCK_NUM; block++)
					if(gpu_map.v.at(block*N + row) != -1) 
						synchronized_map[row] = gpu_map.v.at(block*N + row),
						block=BLOCK_NUM;

			#pragma endregion
			
			#pragma region back substitution
			for(uint function = N-1; function >= 0; --function)
			{
				// Tends to drop off the far end of the array in MSVC, so break just in case
				// Even better - it appears that MSVC is OK with the idea of a unsigned integer having a <0 value
				if(function > N-1 || function < 0) 
					break;

				// Check for the row in which our currently backsubstituted function is
				int fnIdx = functionAt(synchronized_map, function);

				// If function is in "no row", throw UnsolvableException
				// Basically if this happens, the input matrix was never sane in the first place
				if(fnIdx < 0) 
					throw unsex;
				
				// Seed the solution with RHS data
				solution[function] = gpu_rhs.v.at(fnIdx);
				
				// Since we're at x_i, terms x_i+1,...,x_n need to be applied
				for(uint column = N-1; column > function; column--)
				{
					ScalarType multiplier = gpu_matrix.classic_matrix[fnIdx][column]; 	// In ax_i,j - this is the a term
					ScalarType times = solution[column];								// x_c = ...
					
					// Subtract the term which we're considering
					// In Gauss elimination this would need to be subtracted from both sides of the particular equation
					// Since our matrix is not used for anything afterwards, the matrix zeroing is skipped
					solution[function] -= multiplier*times;
				}
				
				// Finally, we're left with an expression of ax_i = ...
				// Dividing by a gives us the actual value of x_i, which is what we want in our solution
				ScalarType ownMultiplier = gpu_matrix.classic_matrix[fnIdx][function];
				// Store the solution
				solution[function] /= ownMultiplier;
			}
			#pragma endregion
			
			#pragma region Free up the OCL resources
			delete cpu_ops;
			
			backend->releaseData(gpu_rhs_handle);
			backend->releaseData(gpu_map_handle);
			backend->releaseData(gpu_N_handle);
			backend->releaseData(gpu_blocksize_handle);
			backend->releaseData(gpu_blocknum_handle);
			backend->releaseData(gpu_mx_h1);
			backend->releaseData(gpu_mx_h2);
			backend->releaseData(gpu_mx_h3);
			
			backend->releaseKernel(Mangler);
			backend->releaseKernel(Resolver);
			#pragma endregion
		}

		/// Function applies reordering to the matrix
		/// @todo Create or find a reasonable bandwidth-limiting reordering algorithm
		void Reordering()
		{

		}

		/// Helper function which compiles the kernels and replaces the internal tags
		/// @remark Tags are a way of circumventing the OpenCL's inability to allocate memory dynamically in a kernel
		void CompileKernels()
		{
			std::string mykernels = "";

			mykernels = util::replaceKernelTag(__pjws__kernels, "--TAG_LOCAL_MAP_SIZE--", N);
			mykernels = util::replaceKernelTag(mykernels, "--TAG_NUMERICAL_ERROR--", NUM_ERR);

			backend->createProgram(mykernels);
		}
	};
}
