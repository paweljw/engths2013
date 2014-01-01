#pragma once

#include <fstream>
#include <map>
#include <vector>
#include <string>

typedef unsigned int uint;

namespace PJWFront
{
	/// The exception to be thrown in case of an unsolvable equation system
	static class UnsolvableException : public std::exception
	{
		virtual const char* what() const throw()
		{
			return "Equation system is probably unsolvable.";
		}
	} unsex;
	
	/// The catch-all custom exception
	/// @deprecated
	static class BadException : public std::exception
	{
		virtual const char* what() const throw()
		{
			return "General badness happened.";
		}
	} badex;
	
	static class NanException : public std::exception
	{
		virtual const char* what() const throw()
		{
			return "Probably QNAN in results.";
		}
	} nanex;
	
	namespace util
	{
		/// Reads a kernel from file (by filename)
		/// @remark I've chosen to use a builtin kernel string, but I'm keeping this here just in case
		/// @param filename Path to the file from which a kernel is to be read
		/// @throws std::exception
		/// @returns Kernel in the form of a std::string.
		inline std::string readKernelFromFile( std::string filename ) 
		{
			std::string filePath = filename;
			std::ifstream file(filePath.c_str(), std::ifstream::in);

			if(!file){
				throw std::exception();
			}

			return std::string(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
		}

		/// Checks the bandwidth of a std-style matrix
		/// @param matrix The matrix to check
		/// @tparam ScalarType The scalar type used in the matrix
		/// @remark The use of typename declarations here might feel excessive, but g++ 4.7 felt really lost without those.
		/// @returns Matrix bandwidth
		template <typename ScalarType>
		inline int MatrixBandwidth(std::vector<typename std::map<unsigned int, ScalarType> > const & matrix)
		{
			int bw = 0;

			for (std::size_t i = 0; i < matrix.size(); i++)
			{
			  uint min_index = matrix.size();
			  uint max_index = 0;
			  for (typename std::map<unsigned int, ScalarType>::const_iterator it = matrix[i].begin();  it != matrix[i].end(); it++)
			  {
				if (it->first > max_index)
				  max_index = it->first;
				if (it->first < min_index)
				  min_index = it->first;
			  }

			  if (max_index > min_index) //row isn't empty
				bw = bw > (max_index - min_index) ? bw : (max_index - min_index); // poniewa¿ std::max nie chce siê kompilowaæ
			}

			return bw;
		}

		/// Checks the bandwidth of a reordered std-style matrix using a reordering vector
		/// @param matrix The matrix to check
		/// @param r The resultant vector of the reordering
		/// @tparam ScalarType The scalar type used in the matrix
		/// @remark The use of typename declarations here might feel excessive, but g++ 4.7 felt really lost without those.
		/// @returns Matrix bandwidth after the reordering
		template <typename ScalarType>
		inline int ReorderedMatrixBandwidth(std::vector<typename std::map<unsigned int, ScalarType> > const & matrix,  std::vector<int> const & r)
		{
			std::vector<int> r2(r.size());
			int bw = 0;

			for (std::size_t i = 0; i < r.size(); i++)
				r2[r[i]] = i;

			for (std::size_t i = 0; i < r.size(); i++)
			{
			  int min_index = matrix.size();
			  int max_index = 0;
			  for (typename std::map<unsigned int, ScalarType>::const_iterator it = matrix[r[i]].begin();  it != matrix[r[i]].end(); it++)
			  {
				if (r2[it->first] > max_index)
				  max_index = r2[it->first];
				if (r2[it->first] < min_index)
				  min_index = r2[it->first];
			  }
			  if (max_index > min_index)
				bw = std::max(bw, max_index - min_index);
			}

			return bw;
		}

		void myReplace(std::string& str, const std::string& oldStr, const std::string& newStr)
		{
		  size_t pos = 0;
		  while((pos = str.find(oldStr, pos)) != std::string::npos)
		  {
		     str.replace(pos, oldStr.length(), newStr);
		     pos += newStr.length();
  		}		
		}

		/// A function to replace a tag (=a particular string) in a kernel string. Just a simple helper, really.
		/// @param kernel The string to perform replacement on
		/// @param tag The string to be replaced
		/// @param value The string that will replace tag
		/// @returns The kernel string after performing the replacement operation
		inline std::string replaceKernelTag(std::string kernel, std::string tag, std::string value)
		{
			myReplace(kernel, tag, value);
			return kernel;
		}
		
		/// A function to replace a tag (=a particular string) in a kernel string with an unsigned int value.
		/// @param kernel The string to perform replacement on
		/// @param tag The string to be replaced
		/// @param value The uint that is to be cast to a string that will replace tag
		/// @returns The kernel string after performing the replacement operation
		inline std::string replaceKernelTag(std::string kernel, std::string tag, uint value)
		{
			char* val = new char[25];
			sprintf(val, "%d", value);
			
			return replaceKernelTag(kernel, tag, std::string(val));	
		}

		/// A function to replace a tag (=a particular string) in a kernel string with a float value.
		/// @param kernel The string to perform replacement on
		/// @param tag The string to be replaced
		/// @param value The float that is to be cast to a string that will replace tag
		/// @returns The kernel string after performing the replacement operation
		template <typename ScalarType>
		inline std::string replaceKernelTag(std::string kernel, std::string tag, ScalarType value)
		{
			char* val = new char[25];
			sprintf(val, "%f", value);
			
			return replaceKernelTag(kernel, tag, std::string(val));	
		}
	}		
}