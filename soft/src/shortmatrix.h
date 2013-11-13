#pragma once

#include "util.h"
#include "oclbackend.h"
// It is so very tempting to remove this altogether.
// But this class cannot be silent.
#include <iostream>

namespace PJWFront
{
	namespace util
	{
		/// A matrix storage type that handles GPU transfers and does a little bit of compression by omission.
		/// It allows for the solver to work because it allocates zero-valued cells where there will be needed by the solver (contrary to typical compressed/coordinate matrix solutions).
		/// This is a middle ground between great compression and ease of coding the kernels, and is potentially a huge to-do.
		/// Given the nature of the compression, it works best with narrow-band matrices.
		/// @author Pawel J. Wal
		/// @tparam ScalarType The type of the scalars used to store data
		template <typename ScalarType>
		class short_matrix
		{
		private:
			/// A backend pointer
			ocl::OCLBackend* backend;
			
			/// Matrix size (we're automatically assuming it's a square matrix).
			uint size;
			
			/// Storage for row offset numbers
			uint* row_offsets;
			
			/// Storage for column offset numbers (not exactly aptly named)
			uint* row_begins;

			/// Number of actual data points (= size of matrix after compression to vector)
			uint DataPoints;
		
			/// The flat-format storage pointer
			ScalarType* flat_matrix;

			/// Whether a recalculation is needed before pushing it to GPU.
			bool NeedsUpdate;

			/// Internal function to regenerate the flat matrix and offset tables
			void Update()
			{
				if(!NeedsUpdate) return;

				#pragma region Size requirements analysis

				// Analysis of the matrix row by row
				for(uint row=0; row < size; row++)
				{
					// Checking which parts of the row can be ommited (are before first significant term)
					// and filling out the offset data on the spot
					for(uint col=0; col<size; col++)
					{
						row_begins[row] = size+1;
						row_offsets[row] = size+1;
						if(classic_matrix[row][col] >= 0.0001)
							row_begins[row] = col,
							row_offsets[row] = (row == 0) ? 0 : ((size-row_begins[row-1]) + (row_offsets[row-1])),
							DataPoints += (size-col),
							col = size;
					}
				}

				#pragma endregion
				
				#pragma region Matrix compression
				
				// Recreating the flat-format vector storage with precisely enough space
				flat_matrix = new ScalarType[DataPoints];

				// A global index so we know where to put our data
				uint global_ix = 0;

				// Copy relevant spots from matrix to their designated spots in the vector
				for(uint row=0; row < size; row++)
					for(uint col=row_begins[row]; col < size; col++)
						flat_matrix[global_ix] = classic_matrix[row][col],
						global_ix++;
				
				#pragma endregion
				
				// Update successful
				NeedsUpdate = false;
			}
			
			
			
		public:
			/// This matrix is sometimes required to be accessible from outside
			/// @todo Code a more clean, OOP solution to this (getters/setters instead of blatant misuse of public visibility)
			ScalarType** classic_matrix;
			
			/// Simple constructor that allocates enough space for the classic square matrix and zeros it out.
			/// Also sets up relevant storages.
			/// @param N Size of the matrix
			/// @param b An OCLBackend pointer
			short_matrix(uint N, ocl::OCLBackend* b)
			{
				size = N;
				classic_matrix = new ScalarType*[N];
				
				for(uint i=0; i<N; i++)
				{
					classic_matrix[i] = new ScalarType[N];
					for(uint j=0; j<N; j++)
						classic_matrix[i][j] = 0;
				}
				DataPoints = 0;
				NeedsUpdate = true;

				row_offsets = new uint[N];
				row_begins = new uint[N];
				
				backend = b;
			}

			/// Empty constructor for compliance
			short_matrix()
			{
			}

			/// Overloaded operator () - basically just a handy getter/setter
			/// @param x Matrix row
			/// @param y Matrix column
			/// @returns A reference to the matrix cell at x, y.
			ScalarType& operator()(uint x, uint y)
			{
				NeedsUpdate = true;
				return classic_matrix[x][y];
			}

			/// Overloaded operator that allows assignment of a matrix (std:: namespace style) to an object of the class.
			/// Basically just a wholesale setter.
			/// @param input An std::vector of uint->ScalarType maps (std:: style matrix).
			/// @throws std::exception
			void operator=(std::vector< std::map< unsigned int, ScalarType> > input)
			{
				if(input.size() > size*size)
				{
					std::cout << "Wrong matrix size (" << input.size() << ">" << size*size << ")" << std::endl;
					throw std::exception();
				}

				for(uint row=0;row<size;row++)
					for(uint col=0;col<size;col++)
						(*this)(row, col) = input[row][col];
						
				NeedsUpdate = true;
			}

			/// Gets the flat-style matrix OCL handle using the backend after checking whether an update is necessary and potentially performing it.
			/// @returns An OpenCL handle for the flat-style matrix.
			cl_mem ocl_flat_handle()
			{
				Update();
				return backend->sendData((void*)flat_matrix, (DataPoints) * sizeof(ScalarType));
			}

			/// Gets the row offsets OpenCL handle.
			/// @returns An OpenCL handle for the row offsets vector.
			/// @todo This could actually send a read-only memory; whether this will shave any time off the sends remains to be explored.
			cl_mem ocl_offsets_handle()
			{
				Update();
				return backend->sendData((void*)row_offsets, size * sizeof(uint));
			}

			/// Gets the column offsets OpenCL handle.
			/// @returns An OpenCL handle for the column offsets vector.
			/// @todo This could actually send a read-only memory; whether this will shave any time off the sends remains to be explored.
			cl_mem ocl_rowfns_handle()
			{
				Update();
				return backend->sendData((void*)row_begins, size * sizeof(uint));
			}
			
			/// Pulls matrix data back from a cl_mem object.
			/// @remark It's better to pass it the handle that it generated itself; the results might be unpredictable if a different handle, even of the same size, is passed int.
			/// @param data The cl_mem handle for the data to be pulled back.
			void pull_data(cl_mem data)
			{
				backend->receiveData(data, flat_matrix, DataPoints * sizeof(ScalarType));
				
				for(int i=0; i<size; i++)
					for(int j=0; j<size; j++)
						classic_matrix[i][j] = 0;
				
				for(int i=0; i<size; i++) // i is a row number now
				{
					uint begin = row_begins[i];
					uint offset = row_offsets[i];
					if(begin > size) continue;
					
					for(int j=begin; j<size; j++)
					{
						classic_matrix[i][j] = flat_matrix[offset+(j-begin)];
					}
				}
			}

			/// A debugging/unit testing function.
			/// @deprecated
			ScalarType * __flat_matrix()
			{
				Update();
				return flat_matrix;
			}

			/// A debugging/unit testing function.
			/// @deprecated
			uint __datapoints()
			{
				Update();
				return DataPoints;
			}

			/// A debugging/unit testing function.
			/// @deprecated
			uint * __rowbegins()
			{
				Update();
				return row_begins;
			}

			/// A debugging/unit testing function.
			/// @deprecated
			uint * __rowoffsets()
			{
				Update();
				return row_offsets;
			}
			
			/// Prints the formatted matrix as it currently is stored in the CPU memory.
			void printMatrix()
			{
				cout << "----------------------------------" << endl;
				cout << "Printing an " << size << "x" << size << " matrix" << endl;
				for(int i=0; i<size; i++)
				{
					cout << "Row " << i << ": \t";
					for(int j=0; j<size; j++)
						std::cout << classic_matrix[i][j] << " ";
					std::cout << std::endl;
				}
			}
		};
	}
}
