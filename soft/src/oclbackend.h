#pragma once

#include <CL/cl.h>
#include <CL/opencl.h>
#include <assert.h>

#include <stdio.h>
#include <string.h>
#include <iostream>
using namespace std;

namespace PJWFront
{
	namespace ocl
	{

		/// A collection of helper functions and storage for certain parts of OpenCL C bindings
		/// @remark Despite being C++/OOP-like, this class errors out C-style with exit() statements instead of C++-style exceptions (potentially todo)
		/// @author Pawel J. Wal
		class OCLBackend
		{
			// Basic platform info
			
			/// OpenCL context
			cl_context context;
			
			/// OpenCL platform
			cl_platform_id platform;
			
			/// OpenCL queue
			cl_command_queue queue;
			
			/// OpenCL device placeholder
			cl_device_id device;

			/// This OpenCL backend will only hold a single program
			cl_program program;

			/// A helper function to get and select an OpenCL platform
			cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID)
			{
				char chBuffer[1024];
				cl_uint num_platforms;
				cl_platform_id* clPlatformIDs;
				cl_int ciErrNum;
				*clSelectedPlatformID = NULL;
				cl_uint i = 0;

				// Get OpenCL platform count
				ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
				if (ciErrNum != CL_SUCCESS)
				{
					//shrLog(" Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
					printf(" Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
					return -1000;
				}
				else
				{
					if(num_platforms == 0)
					{
						//shrLog("No OpenCL platform found!\n\n");
						printf("No OpenCL platform found!\n\n");
						return -2000;
					}
					else
					{
						// if there's a platform or more, make space for ID's
						if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL)
						{
							//shrLog("Failed to allocate memory for cl_platform ID's!\n\n");
							printf("Failed to allocate memory for cl_platform ID's!\n\n");
							return -3000;
						}

						// get platform info for each platform and trap the NVIDIA platform if found
						ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
						printf("Available platforms:\n");
						for(i = 0; i < num_platforms; ++i)
						{
							ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
							if(ciErrNum == CL_SUCCESS)
							{
								printf("platform %d: %s\n", i, chBuffer);
								if(strstr(chBuffer, "NVIDIA") != NULL)
								{
									printf("selected platform %d\n", i);
									*clSelectedPlatformID = clPlatformIDs[i];
									//break;
								}
							}
						}

						// default to zeroeth platform if NVIDIA not found
						if(*clSelectedPlatformID == NULL)
						{
							//shrLog("WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!\n\n");
							//printf("WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!\n\n");
							printf("selected platform: %d\n", 0);
							*clSelectedPlatformID = clPlatformIDs[0];
						}

						free(clPlatformIDs);
					}
				}

				return CL_SUCCESS;
			}

			/// A helper function decoding OpenCL error codes to more useful string representations
			/// @param error The error number
			/// @return String containing error reason flag as cstring
			const char* oclErrorString(cl_int error)
			{
				static const char* errorString[] = {
					"CL_SUCCESS",
					"CL_DEVICE_NOT_FOUND",
					"CL_DEVICE_NOT_AVAILABLE",
					"CL_COMPILER_NOT_AVAILABLE",
					"CL_MEM_OBJECT_ALLOCATION_FAILURE",
					"CL_OUT_OF_RESOURCES",
					"CL_OUT_OF_HOST_MEMORY",
					"CL_PROFILING_INFO_NOT_AVAILABLE",
					"CL_MEM_COPY_OVERLAP",
					"CL_IMAGE_FORMAT_MISMATCH",
					"CL_IMAGE_FORMAT_NOT_SUPPORTED",
					"CL_BUILD_PROGRAM_FAILURE",
					"CL_MAP_FAILURE",
					"",
					"",
					"",
					"",
					"",
					"",
					"",
					"",
					"",
					"",
					"",
					"",
					"",
					"",
					"",
					"",
					"",
					"CL_INVALID_VALUE",
					"CL_INVALID_DEVICE_TYPE",
					"CL_INVALID_PLATFORM",
					"CL_INVALID_DEVICE",
					"CL_INVALID_CONTEXT",
					"CL_INVALID_QUEUE_PROPERTIES",
					"CL_INVALID_COMMAND_QUEUE",
					"CL_INVALID_HOST_PTR",
					"CL_INVALID_MEM_OBJECT",
					"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
					"CL_INVALID_IMAGE_SIZE",
					"CL_INVALID_SAMPLER",
					"CL_INVALID_BINARY",
					"CL_INVALID_BUILD_OPTIONS",
					"CL_INVALID_PROGRAM",
					"CL_INVALID_PROGRAM_EXECUTABLE",
					"CL_INVALID_KERNEL_NAME",
					"CL_INVALID_KERNEL_DEFINITION",
					"CL_INVALID_KERNEL",
					"CL_INVALID_ARG_INDEX",
					"CL_INVALID_ARG_VALUE",
					"CL_INVALID_ARG_SIZE",
					"CL_INVALID_KERNEL_ARGS",
					"CL_INVALID_WORK_DIMENSION",
					"CL_INVALID_WORK_GROUP_SIZE",
					"CL_INVALID_WORK_ITEM_SIZE",
					"CL_INVALID_GLOBAL_OFFSET",
					"CL_INVALID_EVENT_WAIT_LIST",
					"CL_INVALID_EVENT",
					"CL_INVALID_OPERATION",
					"CL_INVALID_GL_OBJECT",
					"CL_INVALID_BUFFER_SIZE",
					"CL_INVALID_MIP_LEVEL",
					"CL_INVALID_GLOBAL_WORK_SIZE",
				};

				const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

				const int index = -error;

				return (index >= 0 && index < errorCount) ? errorString[index] : "";

			}

		public:
		
			/// The default constructor, which will setup the required OpenCL structs
			/// A platform and device will be selected, a context and command queue will be created
			OCLBackend()
			{
				cl_int error;

				// Platform setup
				error = oclGetPlatformID(&platform);
				if (error != CL_SUCCESS) {
					cout << "Error getting platform id: " << oclErrorString(error) << endl;
				   exit(error);
				}

				// Device setup
				error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
				if (error != CL_SUCCESS) {
					cout << "Error getting device ids: " << oclErrorString(error) << endl;
				   exit(error);
				}

				// Context setup
				context = clCreateContext(0, 1, &device, NULL, NULL, &error);
				if (error != CL_SUCCESS) {
					cout << "Error creating context: " << oclErrorString(error) << endl;
					exit(error);
				}

				// CQ setup
				queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
				if (error != CL_SUCCESS) {
					cout << "Error creating command queue: " << oclErrorString(error) << endl;
					exit(error);
				}
			}

			/// Creates and pushes a buffer to the selected device
			/// @param dataptr A voidptr pointing to the data
			/// @param datasize Total size of the data (basically sizeof(type) * amount_of_data)
			/// @param flags cl_mem_flags structure containing the flags to be passed to clCreateBuffer; defaults to read+write memory and copying host pointer.
			/// @return cl_mem typed handle for the GPU buffer
			/// @todo Explore whether a scenario in which an asynchronous data push would make sense exists
			cl_mem sendData(void* dataptr, size_t datasize, cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR)
			{
				cl_int error;

				// Create the GPU buffer
				// Since this does not, in fact, push the data to the device, I will never understand
				// Why does it require a parameter that points to the actual host data structure.
				cl_mem buffer = clCreateBuffer(context, flags, datasize, dataptr, &error);
				if(error != CL_SUCCESS)
				{
					cout << "Error sending data in CCB: " << oclErrorString(error) << endl;
					exit(error);
				}
				
				// Push the data to the GPU in a blocking way
				error = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, datasize, dataptr, 0, NULL, NULL);
				if(error != CL_SUCCESS)
				{
					cout << "Error sending data in EWB: " << oclErrorString(error) << endl;
					exit(error);
				}
				
				// Return the buffer
				return buffer;
			}

			/// Pulls the data from GPU to a host buffer
			/// @param data The data handle
			/// @param dataptr Voidptr-cast host buffer pointer
			/// @param datasize Amount of data to be read and stored
			/// @todo Explore non-blocking read scenarios
			void receiveData(cl_mem data, void* dataptr, size_t datasize)
			{
				cl_int error = clEnqueueReadBuffer(queue, data, CL_TRUE, 0, datasize, dataptr, 0, NULL, NULL); 
				if(error != CL_SUCCESS)
				{
					cout << "Error reading data [" << datasize << " bits]: " << oclErrorString(error) << endl;
					exit(error);
				}
			}

			/// Creates (compiles) an OpenCL program from a kernel-string. The program is then stored internally in this class.
			/// This class cannot store more than one program at a time, it can, however, store more than one kernel.
			/// @param progstr OpenCL program as string
			void createProgram(std::string progstr)
			{
				// I vaguely remember someone telling me that this cast wasn't really necessary
				const char * c = progstr.c_str();

				cl_int error = 0;

				// Create the program from source
				program = clCreateProgramWithSource(context, 1, &c, NULL, &error);
				if (error != CL_SUCCESS) {
					cout << "Error creating program from src: " << oclErrorString(error) << endl;
					exit(error);
				}

				// Build the program
				error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
				
				// If the build failed and the __SOLVERDEBUG flag is set, the build log will be printed
				// Otherwise, just the error code will be printed
				if (error != CL_SUCCESS) {
					cout << "Error compiling OCL program: " << oclErrorString(error) << endl;
#if defined(__SOLVERDEBUG) || defined(__SOLVERTIMING)
					char* build_log;
					size_t log_size;
					// First call to know the proper size
					clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
					build_log = new char[log_size+1];
					// Second call to get the log
					clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
					build_log[log_size] = '\0';
					cout << build_log;
					delete[] build_log;
#endif
					exit(error);
				}
			}

			/// Get a OpenCL kernel handle by kernel name
			/// @param name Name of the kernel to get ahold of
			/// @returns OpenCL kernel handle
			cl_kernel getNamedKernel(std::string name)
			{
				cl_int error = 0;
				
				// Create the kernel object with some error handling just in case
				cl_kernel ret = clCreateKernel(program, name.c_str(), &error);
				if (error != CL_SUCCESS) {
					cout << "Error creating named kernel: " << oclErrorString(error) << endl;
					exit(error);
				}

				size_t kernel_work_group_size;
				clGetKernelWorkGroupInfo(ret, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_work_group_size, NULL);
				
				return ret;
			}

			/// Set a cl_mem kernel argument for a particular kernel
			/// @param kernel Kernel handle object
			/// @param arg Which kernel argument to set (by number)
			/// @param data The data handle to be set as argument
			/// @todo Explore whether the templated attempt does actually get called instead of this one, as this might warrant a deprecated tag
			void arg(cl_kernel kernel, unsigned int arg, cl_mem data)
			{
				cl_int error = clSetKernelArg(kernel, arg, sizeof(cl_mem), &data);
				if(error != CL_SUCCESS)
				{
					cout << "Error setting arg " << arg << ": " << oclErrorString(error) << endl;
					exit(error);
				}
			}
			
			/// Set a generic kernel argument for a particular kernel
			/// @param kernel Kernel handle object
			/// @param arg Which kernel argument to set (by number)
			/// @param data The data to be set as argument
			/// @tparam T type of the data being set up
			template<typename T>
			void arg(cl_kernel kernel, unsigned int arg, T data)
			{
				cl_int error = clSetKernelArg(kernel, arg, sizeof(T), &data);
				if(error != CL_SUCCESS)
				{
					cout << "Error setting _arg " << arg << " [" << sizeof(T) << "]: " << oclErrorString(error) << endl;
					exit(error);
				}
			}

			/// Put a kernel into the command queue
			/// @param kernel The kernel to enqueue
			/// @param local_work_size The local block size
			/// @param global_work_size The global work size (total number of spawned threads)
			void enqueueKernel(cl_kernel kernel, size_t local_work_size, size_t global_work_size)
			{
				cl_int error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
				if(error != CL_SUCCESS)
				{
					cout << "Error enqueueing kernel: " << oclErrorString(error) << endl;
					exit(error);
				}
			}

			/// Put a kernel into the command queue while attaching a traceable and profilable event
			/// @param kernel The kernel to enqueue
			/// @param local_work_size The local block size
			/// @param global_work_size The global work size (total number of spawned threads)
			/// @returns A cl_event object storing the attached event
			cl_event enqueueEventKernel(cl_kernel kernel, size_t local_work_size, size_t global_work_size)
			{
				cl_event event;

				cl_int error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &event);
				if(error != CL_SUCCESS)
				{
					cout << "Error enqueueing kernel: " << oclErrorString(error) << endl;
					exit(error);
				}

				return event;
			}

			/// Releases (destroys) a named kernel object
			/// @param kernel Kernel to be released
			void releaseKernel(cl_kernel kernel)
			{
				clReleaseKernel(kernel);
			}

			/// Releases (destroys) a cl_mem data handle
			/// @param data Data handle to be released
			void releaseData(cl_mem data)
			{
				clReleaseMemObject(data);
			}

			/// Blocks until the commands in the queue have been completed, successfully or otherwise.
			void finish()
			{
				cl_int error = clFinish(queue);
				if(error != CL_SUCCESS)
				{
					cout << "Error finishing non-named queue: " << oclErrorString(error) << endl;
					exit(error);
				}
				
			}
			
			void finish(std::string what)
			{
				cl_int error = clFinish(queue);
				if(error != CL_SUCCESS)
				{
					cout << "Error finishing queue [" << what.c_str() << "]: " << oclErrorString(error) << endl;
					exit(error);
				}
				
			}
			
			/// Blocks until the command attached to the event has finished, then returns its elapsed time.
			/// @param event The cl_event to wait for.
			/// @returns Total elapsed time for the event in seconds.
			double timedFinish(cl_event event)
			{
				finish("timing event");
				clWaitForEvents(1 , &event);
				cl_ulong time_start, time_end;
				double total_time;

				cl_int err;

				err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
				if(err != CL_SUCCESS)
				{
					cout << "Error timing kernel [start]: " << oclErrorString(err) << "[" << err << "]" << endl;
					exit(err);
				}

				err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
				if(err != CL_SUCCESS)
				{
					cout << "Error timing kernel [stop]: " << oclErrorString(err) << "[" << err << "]" << endl;
					exit(err);
				}

				total_time = time_end - time_start;
				total_time /= 1000000000.0;
				return total_time;
			}

			/// General clean-up destructor
			~OCLBackend()
			{
				clReleaseCommandQueue(queue);
				clReleaseContext(context);
			}
		};
	}
}

