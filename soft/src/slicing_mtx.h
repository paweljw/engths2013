#pragma once

#include "util.h"
#include "oclbackend.h"
#include <iostream>

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace std;
using namespace boost::numeric::ublas;

namespace PJWFront
{
	namespace util
	{
		template <typename ScalarType>
		class slicing_matrix
		{
		private:
			// Wskaünik na backend
			ocl::OCLBackend* backend;
			
			// Rozmiar macierzy - zakladamy kwadratowa
			uint size;

			// Rozmiar slice'a - zakladamy ze slice'y sa slice_height x size
			uint slice_height;

			// Ile jest slice'ow
			uint _slices;
		
			cl_mem cache;

			compressed_matrix<ScalarType> m;

			uint* slice_left;
			uint* slice_right;

		public:
			std::vector< std::map< unsigned int, ScalarType> > classic_matrix;
			slicing_matrix() { }

			slicing_matrix(uint _size, uint _slice_height, ocl::OCLBackend* b)
			{
				size = _size;
				slice_height = _slice_height;

				// Ustalenie iloúci slice'Ûw
				_slices = 0;
				while(_slices * slice_height < size) _slices++;

				m = compressed_matrix<ScalarType>(size, size);

				backend = b;
				
				// cout << _slices  << " slices" << endl;

				slice_left = new uint[_slices];
				slice_right = new uint[_slices];

				for(uint i = 0; i < _slices; i++)
				{
					slice_left[i] = size;
					slice_right[i] = 0;
				}
			}

			uint slices()
			{
				return _slices;
			}

			uint which_slice(uint x)
			{
				return (int)x / slice_height;
			}

			void set(uint x, uint y, ScalarType val)
			{
				if(m(x, y) != val)
					m(x, y) = val;

				uint slice = which_slice(x);

				slice_left[slice] = y < slice_left[slice] ? y : slice_left[slice];
				slice_right[slice] = slice_right[slice] < y ? y : slice_right[slice];
			}

			uint slice_leftmost(uint slice)
			{
				return slice_left[slice];
			}

			uint slice_rightmost(uint slice)
			{
				return slice_right[slice];
			}

			uint slice_width(uint slice)
			{
				return (slice_right[slice] - slice_left[slice]) + 1;
			}

			void add(uint x, uint y, ScalarType val)
			{
				val += m(x, y);
				set(x, y, val);
			}

			ScalarType get(uint x, uint y)
			{
				return m(x, y);
			}

			cl_mem get_slice(uint slice_number, bool _cache = false)
			{
				if(_cache) return cache;

				ScalarType *flat_slice = new ScalarType[ slice_width(slice_number) * slice_height];

				uint global_ix = 0;

				// rows
				for(uint i = slice_number * slice_height; i < (slice_number + 1) * slice_height; i++)
				{
					// cols
					for(uint j = slice_leftmost(slice_number); j <= slice_rightmost(slice_number); j++)
					{
						if(i >= size) // jesli spadamy z macierzy
							flat_slice[global_ix] = 0;
						else // w innym przypadku
							flat_slice[global_ix] = m(i, j);

						//cout << flat_slice[global_ix] << " ";

						global_ix++;
					}
					//cout << endl;
				}

				cache = backend->sendData((void*)flat_slice, (slice_width(slice_number) * slice_height) * sizeof(ScalarType));

				backend->finish();
				delete [] flat_slice;

				return cache;
			}

			void pull_slice_from_cache(uint slice_number)
			{
				pull_slice(slice_number, cache);
			}

			void pull_slice(uint slice_number, cl_mem data)
			{
				ScalarType *flat_slice = new ScalarType[slice_width(slice_number) * slice_height];

				backend->receiveData(data, flat_slice, (slice_width(slice_number) * slice_height) * sizeof(ScalarType));

				uint global_ix = 0;

				// rows
				for(uint i = slice_number * slice_height; i < (slice_number + 1) * slice_height; i++)
				{
					// cols
					for(uint j = slice_leftmost(slice_number); j <= slice_rightmost(slice_number); j++)
					{
						if(i >= size) // jesli spadamy z macierzy
							break;
						else // w innym przypadku
							set(i, j, flat_slice[global_ix]);

						global_ix++;
					}
				}

				delete [] flat_slice;
			}

			void printMatrix()
			{
				cout << "----------------------------------" << endl;
				cout << "Printing an " << size << "x" << size << " matrix" << endl;
				for(uint i=0; i<size; i++)
				{
					cout << "Row " << i << ": \t";
					for(uint j=0; j<size; j++)
						std::cout << m(i, j) << " ";
					std::cout << std::endl;
				}
			}
		}; // class
	} // ns util
} // ns pjwfront
