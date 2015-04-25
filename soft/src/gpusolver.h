#pragma once

#include "util.h"
#include "oclbackend.h"
#include "slicing_mtx.h"
#include "slicingvector.h"
#include "oclvector.h"

#include <boost/timer/timer.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace PJWFront
{

	class comp_nonzero_def : public boost::numeric::ublas::compressed_matrix<int>
	{
    int def;
	public:
		comp_nonzero_def() { }
		comp_nonzero_def( int s1, int s2 )
			: boost::numeric::ublas::compressed_matrix<int>(s1,s2) , def(-1)
    {

    }
    void setDefault( int d ) { def = d; }
    int value( int i, int j )
    {
		//cout << "Enter value " << i << ", " << j << endl;
        typedef boost::numeric::ublas::compressed_matrix<int>::iterator1 it1_t;
        typedef boost::numeric::ublas::compressed_matrix<int>::iterator2 it2_t;
        for (it1_t it1 = begin1(); it1 != end1(); it1++)
        {
            if( it1.index1() <  i )
                continue;
            if( it1.index1() > i ) {
                return def;
            }
            for (it2_t it2 = it1.begin(); it2 != it1.end(); it2++)
            {
                if( it2.index2() < j )
                    continue;
                if( it2.index2() == j )
                    return *it2;
                if( it2.index2() > j )
                    return def;
            }
        }
        return def;
		}
    };

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
		util::slicing_matrix<ScalarType>	gpu_matrix;
		/// CPU-to-GPU RHS storage structure
		util::slicing_vector<ScalarType>	gpu_rhs;
		/// CPU-to-GPU mapping storage structure
		comp_nonzero_def cpu_map;

		/// Size of an NxN matrix
		uint N;

		uint slices;

		uint slice_size;

		unsigned int *synchronized_map;

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

		inline void ReduceRows(const unsigned int original, const unsigned int offender, const unsigned int function)
		{
			ScalarType upper = gpu_matrix.get(original, function);
			ScalarType lower = gpu_matrix.get(offender, function);

			ScalarType multiplier = upper / lower;
			multiplier *= (double) -1.0f;

			for(unsigned int i = function; i < N; i++)
			{
				if(i==function)
					gpu_matrix.set(original, i, 0.0f);
				else
				{
					if(gpu_matrix.get(offender, i) == 0)
						continue;

					ScalarType byval = gpu_matrix.get(offender, i) * multiplier;

					gpu_matrix.add(original, i, byval);
				}
		}

		gpu_rhs.add(original, gpu_rhs.get(offender) * multiplier);
	}

	public:
		unsigned long fmad;
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

		uint mtx_offset;

		/// compat
		GPUFrontal()
		{
		}

		// mtx_offset is part_num times part_size
		GPUFrontal(int size, uint _LWS = 192, uint _GWS = 0, int pts = 0, int dts = 0, uint _mtx_offset = 0)		
		{
			mtx_offset = _mtx_offset;

			// Initialize OpenCL backend
			backend = new ocl::OCLBackend(pts, dts);

			//cout << "Set up backend" << endl;

			// Remember matrix size
			N = size;

			//cout << "Set size, lol" << endl;

			// Remember numerical error
			NUM_ERR = 0.0000000001;

			//cout << "Set numerical error, lol" << endl;

			// Nic tu nie cwaniakujemy
			LWS = _LWS;

			// Tu te� nie
			GWS = _GWS;

			///cout << "Set LWS and GWS, lol" << endl;

			fmad = (unsigned long)0;

			//cout << "Set fmad, lol" << endl;
			
			BLOCK_NUM = GWS/LWS;

			//cout << "Set block_num, lol" << endl;

			//cout << "BN: " << BLOCK_NUM << ", GWS: " << GWS << ", LWS: " << LWS << endl;

			slice_size = GWS;

			// Inicjalizacja kawa�kuj�cej macierzy; slice ma mie� taki rozmiar jak GWS
			gpu_matrix = util::slicing_matrix<ScalarType>(N, slice_size, backend);

			//cout << "slicing matrix done" << endl;

			slices = gpu_matrix.slices();

			//cout << "Done" << endl;

			// A niech tam, whatever
			gpu_rhs = util::slicing_vector<ScalarType>(N, slice_size, backend);

			//cout << "Done 2" << endl;

			// To akurat ma sens, tylko trzeba b�dzie tego u�ywa� mocno dooko�a
			cpu_map = comp_nonzero_def(slices, N);
			cpu_map.setDefault(-1);

			//cout << "Done 3" << endl;

			// To na piej
			solution = std::vector<ScalarType> (N);

			//cout << "Done" << endl;

			// Niech to zostanie, bo nie pami�tam co tu si� dzieje
			CompileKernels();

			//cout << "Aww :<" << endl;
		}

		/// Setter for matrix values
		/// @param row Matrix row
		/// @param col Matrix column
		/// @param val The value to store at given coordinates
		void set(int row, int col, ScalarType val)
		{
			gpu_matrix.set(row, col, val);
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
//			boost::timer::auto_cpu_timer t("Solution took %t s CPU, %w s real");

			#pragma region OCL memory setup

			 // cout << "start" << endl;

			// Kernele zostaj�
			cl_kernel Slicer = backend->getNamedKernel("Slicer");
			cl_kernel Resolver = backend->getNamedKernel("Resolver");

			// M te� si� przyda
			cl_mem gpu_M_handle = backend->sendData(&slice_size, sizeof(uint));

			backend->arg(Slicer, 1,  gpu_M_handle);

			cl_mem gpu_blocknum_handle = backend->sendData(&BLOCK_NUM, sizeof(uint));
			backend->arg(Resolver, 4, gpu_blocknum_handle);

			uint ss = slice_size * slices;
			cl_mem gpu_ss_handle = backend->sendData(&ss, sizeof(uint));
			backend->arg(Resolver, 6, gpu_ss_handle);

			unsigned int cpu_ops = 0; // tym si� przejmujemy ju� tylko na cpu

			ocltime = 0.0f;
			#pragma endregion

			#pragma region Solution loop

			//cout << "mem setup" << endl;

			do
			{
				cpu_ops = 0;
				//cout << "cpu_ops is definitely " << cpu_ops << endl;
				int last_slice = -1;
				bool new_slice = true;


				for(int slice = 0; slice < slices; slice++)
				{
					//cout << ">";

					if(last_slice != slice)
					{
						last_slice = slice;
						new_slice = true;
					}

					if(gpu_matrix.slice_width(slice) == 0) continue;

					//cout << "slice" << endl;
					cl_mem slice_matrix = gpu_matrix.get_slice(slice, !new_slice);
					//cout << "slice2" << endl;
					cl_mem slice_rhs = gpu_rhs.get_slice(slice, !new_slice);
					//cout << "slices" << endl;
					util::ocl_vector<int> map_slicer = util::ocl_vector<int>((gpu_matrix.slice_width(slice)) * BLOCK_NUM, backend);
					map_slicer.fill(-1.0f);
					cl_mem map_slicer_handle = map_slicer.ocl_handle(!new_slice);

					cl_mem gpu_N_handle;

					if(new_slice)
					{
						uint ln = gpu_matrix.slice_width(slice);
						gpu_N_handle = backend->sendData(&ln, sizeof(uint));

						backend->arg(Slicer, 0,  gpu_N_handle);
						backend->arg(Slicer, 2, slice_matrix);
						backend->arg(Slicer, 3, slice_rhs);
						backend->arg(Slicer, 4, map_slicer_handle);
						backend->arg_local<int>(Slicer, 5, gpu_matrix.slice_width(slice));
					}

					uint *flops = new uint;
					*flops = 0;

					cl_mem gpu_flops = backend->sendData(flops, sizeof(uint));

					backend->arg(Slicer, 6, gpu_flops);

					cl_event slicer_event = backend->enqueueEventKernel(Slicer, LWS, GWS);
                    ocltime += backend->timedFinish(slicer_event);

					uint *local_ops = new uint;

					*local_ops = 0;

					cl_mem gpu_local_ops = backend->sendData(local_ops, sizeof(uint));

					if(new_slice)
					{
						backend->arg(Resolver, 0,  gpu_N_handle);
						backend->arg(Resolver, 1, slice_matrix);
						backend->arg(Resolver, 2, slice_rhs);
						backend->arg(Resolver, 3, map_slicer_handle);
					}

					backend->arg(Resolver, 7, gpu_flops);
					backend->arg(Resolver, 5, gpu_local_ops);

					cl_event resolver_event = backend->enqueueEventKernel(Resolver, LWS, GWS);
                    ocltime += backend->timedFinish(resolver_event);

					backend->receiveData(gpu_local_ops, local_ops, sizeof(uint));
					backend->releaseData(gpu_local_ops);

					backend->receiveData(gpu_flops, flops, sizeof(uint));
					backend->releaseData(gpu_flops);

					fmad += (unsigned long)flops;

					if(*local_ops > 0)
					{
						//cout << "replaying slice" << endl;
						slice--;
						new_slice = false;
					} else {
						gpu_matrix.pull_slice(slice, slice_matrix);
						gpu_rhs.pull_slice(slice, slice_rhs);
						map_slicer.pull_data(map_slicer_handle);

						backend->releaseData(slice_matrix);
						backend->releaseData(slice_rhs);
						backend->releaseData(map_slicer_handle);
						backend->releaseData(gpu_N_handle);

						uint height_compensation = slice * slice_size;

						// petla 0..width
						// #pragma omp parallel for
//#pragma omp parallel for
						//for(int i=0; i<N; i++) cpu_map.at(slice, i) = -1;
//#pragma omp barrier
						for(int row = 0; row < gpu_matrix.slice_width(slice); row++)
						{
							// globalny indeks standardowy = slice * N + wiersz
							// globalny indeks wymaga dodatkowej korekty o pominiete funkcje, ergo
							uint global_ix = row + gpu_matrix.slice_leftmost(slice);

							for(uint block=0; block<BLOCK_NUM; block++)
							{
								uint local_ix = block * gpu_matrix.slice_width(slice) + row;

								if(map_slicer.get(local_ix) != -1)
								{
									// dla mapy globalnej powinien byc to juz wlasciwy row
									cpu_map(slice, global_ix) = map_slicer.get(local_ix) + height_compensation;
									block=BLOCK_NUM;
								}
							}
						}

					}
					//cout << "." << endl;
				}

				//cout << "*" << endl;
				#pragma omp barrier
				#pragma omp parallel for
				for(int row = 0; row < N; row++) // petla po wierszach 0..N
					{

					uint slc = gpu_matrix.which_slice(row);
					if(gpu_matrix.slice_width(slc) == 0) continue;

						int first = -1;
						int function = -1;

						for(uint block=0; block<slices; block++) // petla po slice'ach dla wiersza
						{
							// if(gpu_matrix.slice_width(block) == 0) continue;
							//cout << "for slice " << block << endl;
							if(cpu_map.value(block, row) != -1)
							{
							//	cout << "Enter loop" << endl;
								// te koordynaty juz sa globalne
								if(first == -1){
							//		cout << "Went here" << endl;
									first = cpu_map.value(block, row);
									// cout << "first is now " << first << endl;
									function = row;
									continue;
								} else {
									int thisRow = cpu_map.value(block, row);
									#pragma omp critical
									{
										// cout << "reducing row " << thisRow << " offending " << first << endl;
										ReduceRows(thisRow, first, function);
										cpu_ops++;
									}
								}
							}
						}
					//cout << endl;
					}
 				// cout << endl;
				cout << cpu_ops << endl;
				// gpu_matrix.printMatrix();
			} while(cpu_ops > 0);
#pragma omp barrier
//			gpu_matrix.printMatrix();

			#pragma endregion

		//	printf("Time in kernels: %0.9f s\n", ocltime);

			fmads = fmad / ocltime;

		//	printf("Achieved FMAD/s: %0.1f\n", fmads);

//			cout << "Assembling solution" << endl;
			// Local storage for the synchronized (from multiple block maps) map
			synchronized_map = new unsigned int[N];


// cout << "Print RHS here on one line" << endl << "%" << endl;
gpu_rhs.print_data();
cout << "%" << endl;
gpu_matrix.printMatrix(mtx_offset);
cout << "%" << endl;
// the row here is actually the function (y component), so only offset the x - physical row
			for(uint row=0; row<N; row++)
				for(uint block=0; block<slices; block++)
					if(cpu_map.value(block, row) != -1)
					{
						synchronized_map[row] = cpu_map.value(block, row);
						block=slices;
						cout << row << ":" << mtx_offset + synchronized_map[row] << " ";
					}

			#pragma endregion

// what we would like here, is for the map and matrix to be available to the loader


// removing backsubstitution for master thesis; not necessary here

/*
			#pragma region back substitution
			for(uint function = N-1; function >= 0; --function)
			{
				if(!(function % 1000)) cout << function;
				else if(!(function % 100)) cout << ".";
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
				solution[function] = gpu_rhs.get(fnIdx);

				uint begat = gpu_matrix.slice_rightmost(gpu_matrix.which_slice(fnIdx));

				// Since we're at x_i, terms x_i+1,...,x_n need to be applied
				#pragma omp parallel for
				for(int column = begat; column > function; column--)
				{
					ScalarType multiplier = gpu_matrix.get(fnIdx, column); 	// In ax_i,j - this is the a term
					ScalarType times = solution[column];								// x_c = ...

					// Subtract the term which we're considering
					// In Gauss elimination this would need to be subtracted from both sides of the particular equation
					// Since our matrix is not used for anything afterwards, the matrix zeroing is skipped
					#pragma omp atomic
					solution[function] -= multiplier*times;
				}

				// Finally, we're left with an expression of ax_i = ...
				// Dividing by a gives us the actual value of x_i, which is what we want in our solution
				ScalarType ownMultiplier = gpu_matrix.get(fnIdx, function);
				// Store the solution
				solution[function] /= ownMultiplier;
			}
			#pragma endregion
		*/
			#pragma region Free up the OCL resources

			backend->releaseData(gpu_M_handle);
			backend->releaseData(gpu_ss_handle);

			backend->releaseKernel(Slicer);
			#pragma endregion

			cout << endl;
		}

		/// Helper function which compiles the kernels
		void CompileKernels()
		{
			std::string mykernels = util::readKernelFromFile("twokernels.cl");

			backend->createProgram(mykernels);
		}
	};
}
