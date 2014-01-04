#include "util.h"
#include <iostream>

using namespace std;

namespace PJWFront
{
namespace util
{
	/// An std::vector wrapper class, enabling easy (=mindless) transfer of std::vector to the GPU
	/// @remark The wrapper approach was chosen because it is apparently a poor practice to override std:: namespace components!
	/// @tparam ScalarType The scalar type that will be used for storage
	/// @author Pawel J. Wal
	template <typename ScalarType>
	class slicing_vector
	{
	public:
		/// The underlying vector
		std::vector<ScalarType> v;

		cl_mem cache;

		/// A pointer to custom backend type (this facilitates the transfers)
		ocl::OCLBackend* backend;

		/// Size of the data
		uint size;
		
		uint slices;

		uint slice_size;

		/// Empty constructor for compliance
		slicing_vector()
		{
		}

		/// Constructs the class zerofilling the data storage
		/// @param N size of the vector
		/// @param b OCLBackend pointer
		slicing_vector(int N, uint _slice_size, ocl::OCLBackend* b)
		{
			//cout << "Vector ctor enter" << endl;
			v = std::vector<ScalarType>(N);
			size = N;
			zerofill();
			backend = b;

			slice_size = _slice_size;

			slices = 0;
			while(slices * slice_size < size) slices++;
			//cout << "Vector got " << slices  << " slices" << endl;
		}

		/// Simple setter
		/// @param p Store where
		/// @param val Store what
		void put(uint p, ScalarType val)
		{
			v.at(p) = val;
		}

		void add(uint p, ScalarType val)
		{
			val += get(p);
			put(p, val);
		}

		ScalarType get(uint p)
		{
			return v.at(p); 
		}

		/// Stores zeroes throughout the vector
		void zerofill()
		{
			for(uint i=0; i<v.size(); i++)
				v.at(i) = (ScalarType)0.0f;
		}

		/// Allows to extract the OCL data handle
		/// @remark Note that every time this is used, a new handle will be created and the data sent again
		cl_mem get_slice(uint slice, bool _cache = false)
		{
			if(_cache) return cache;

			ScalarType* temp = new ScalarType[slice_size];

			uint global_ix = 0;

			for(uint i = slice_size * slice; i < slice_size * (slice+1); i++)
			{
				if(i >= size)
					temp[global_ix] = 0;
				else
					temp[global_ix] = v.at(i);

				global_ix++;
			}

			cache = backend->sendData(temp, slice_size*sizeof(ScalarType));
			return cache;
		}
		
		void pull_slice_from_cache(uint slice_number)
		{
			pull_slice(slice_number, cache);
		}

		/// Allows to fill the vector again after operations have been done on the GPU.
		/// It is best to pass it the handle that was generated by it; other handles might have unpredictable behavior even if the size is the same.
		void pull_slice(uint slice_number, cl_mem data)
		{
			ScalarType* temp = new ScalarType[slice_size];
			backend->receiveData(data, temp, slice_size*sizeof(ScalarType));
			
			uint global_ix = 0;

			for(uint i = slice_size * slice_number; i < slice_size * (slice_number+1); i++)
			{
				if(i >= size)
					break;
				else
					v.at(i) = temp[global_ix];

				global_ix++;
			}
		}
		
		void print_data()
		{
			for(uint i=0; i<size; i++)
			{
				cout << v.at(i) << " ";
			}
			cout << endl;
		}

		~slicing_vector()
		{
			std::vector<ScalarType>().swap(v);
			//backend->releaseData(cache);
		}
	};
}
}
