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
	class ocl_vector
	{
	public:
		/// The underlying vector
		std::vector<ScalarType> v;
		
		/// The C-array pointer which is used for transfers
		ScalarType* temp;

		/// A pointer to custom backend type (this facilitates the transfers)
		ocl::OCLBackend* backend;

		cl_mem cache;

		/// Size of the data
		uint size;
		
		/// Empty constructor for compliance
		ocl_vector()
		{
		}

		/// Constructs the class zerofilling the data storage
		/// @param N size of the vector
		/// @param b OCLBackend pointer
		ocl_vector(int N, ocl::OCLBackend* b)
		{
			v = std::vector<ScalarType>(N);
			temp = new ScalarType[N];
			size = N;
			zerofill();
			backend = b;
		}

		/// Constructs the class with predefined data
		/// @param N size of the vector
		/// @param vals Pointer to C-array of ScalarType, assumed to be of size N
		/// @param b OCLBackend pointer
		ocl_vector(int N, ScalarType* vals, ocl::OCLBackend* b)
		{
			v = std::vector<ScalarType>(N);
			
			backend = b;
			
			temp = new ScalarType[N];
			
			for(uint i = 0; i < N; i++)
				v.at(i) = vals(i);
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

		void fill(ScalarType value)
		{
				for(uint i=0; i<v.size(); i++)
				v.at(i) = (ScalarType)value;
		}

		/// Allows to extract the OCL data handle
		/// @remark Note that every time this is used, a new handle will be created and the data sent again
		cl_mem ocl_handle(bool _cache = false)
		{
			if(_cache) return cache;
			for(unsigned int i=0; i<size; i++)
				temp[i] = v.at(i);
				
			cache = backend->sendData(temp, size*sizeof(ScalarType));
			return cache;
		}

		void pull_data_from_cache()
		{
			pull_data(cache);
		}
		
		/// Allows to fill the vector again after operations have been done on the GPU.
		/// It is best to pass it the handle that was generated by it; other handles might have unpredictable behavior even if the size is the same.
		void pull_data(cl_mem data)
		{
			backend->receiveData(data, temp, size*sizeof(ScalarType));
			
			for(uint i=0; i<size; i++)
				v.at(i) = temp[i];
		}
		
		void print_data()
		{
			for(uint i=0; i<size; i++)
			{
				cout << v.at(i) << " ";
			}
			cout << endl;
		}

		~ocl_vector()
		{
			std::vector<ScalarType>().swap(v);
		//	backend->releaseData(cache);
		}
	};
}
}
