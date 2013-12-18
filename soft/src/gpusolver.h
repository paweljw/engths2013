#pragma once

#include "util.h"
#include "oclbackend.h"
#include "flatmatrix.h"
#include "oclvector.h"
#include <cfloat>

namespace PJWFront
{
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
		util::flat_matrix<ScalarType>	gpu_matrix;
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
			gpu_matrix = util::flat_matrix<ScalarType>(N, backend);
			cpu_matrix = std::vector< std::map< unsigned int, float> > (N*N);
			gpu_rhs = util::ocl_vector<ScalarType>(N, backend);
			gpu_map = util::ocl_vector<int>(N, backend);
			solution = std::vector<ScalarType> (N);
			
			// Precompile the kernels
			CompileKernels();

			// MAP SETUP!!!
			
			for(uint i=0; i<N; i++)
				gpu_map.v.at(i) = -1;
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

			// Set up kernel object
			cl_kernel Mangler = backend->getNamedKernel("Mangler");
						
			// Generate all the required data handles for kernel argument
			// Note: PCI-e data transfer happening below
			cl_mem gpu_rhs_handle = gpu_rhs.ocl_handle();			
			cl_mem gpu_map_handle = gpu_map.ocl_handle();
			
			cl_mem gpu_N_handle = backend->sendData(&N, sizeof(uint));

			cl_mem gpu_mx_h1 = gpu_matrix.ocl_flat_handle();

			unsigned int *cpu_flops = new unsigned int;
			*cpu_flops = 0;

//			cl_mem gpu_flops = backend->sendData((void*)cpu_flops, sizeof(unsigned int));

			// Note: PCI-e data transfer happening above
			
			/*
			0. __global float* dataMatrix,						// Flat-format data matrix as generated by short_matrix
			1. __global float* dataRhs,							// RHS vector, typical format
			2. __global unsigned int* map,						// Global map, flat format w/o compression
			3. __global const unsigned int N,					// Matrix size (for cases where GWS > N, limiting for loops)
			*/

			// Set up parameters for first kernel; refer to above comment for expected values
			backend->arg(Mangler, 0,  gpu_mx_h1);			
			backend->arg(Mangler, 1,  gpu_rhs_handle);
			backend->arg(Mangler, 2,  gpu_map_handle);
			backend->arg(Mangler, 3,  gpu_N_handle);
			
#ifdef __SOLVERDEBUG
			gpu_matrix.printMatrix();
#endif

			ocltime = 0.0f;
			#pragma endregion

			#pragma region Solution loop
			
			// Enqueue first kernel and let it finish
			// Time first kernel
			cl_event mangler_event = backend->enqueueEventKernel(Mangler, LWS, GWS);
			ocltime += backend->timedFinish(mangler_event);

#ifdef __SOLVERDEBUG
			gpu_matrix.pull_data(gpu_mx_h1);
			gpu_matrix.printMatrix();
				
			cout << "---" << endl;
				
			cout << "MAP: ";
			gpu_map.pull_data(gpu_map_handle);
			for(uint i=0; i<N; i++) // row
			{
				cout << gpu_map.v.at(i) << " ";
			}
			cout << endl;
				
			cout << "RHS: ";
			gpu_rhs.pull_data(gpu_rhs_handle);
			for(uint i=0; i<N; i++) // row
			{
				cout << gpu_rhs.v.at(i) << " ";
			}
			cout << endl;
#endif					

			#pragma endregion

			printf("Time in kernels: %0.9f s\n", ocltime);
			
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
				synchronized_map[row] = gpu_map.v.at(row);


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
			
			backend->releaseData(gpu_rhs_handle);
			backend->releaseData(gpu_map_handle);
			backend->releaseData(gpu_N_handle);
			backend->releaseData(gpu_mx_h1);
			
			backend->releaseKernel(Mangler);
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

			mykernels = util::readKernelFromFile("simplekernels.cl");

			backend->createProgram(mykernels);
		}
	};
}
