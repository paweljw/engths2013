#include "gpusolver.h"
#include "oclbackend.h"
#include "util.h"

int main()
{
	PJWFront::ocl::OCLBackend oclbackend;

	std::string kernels = PJWFront::util::replaceKernelTag(PJWFront::__pjws__kernels, "--TAG_LOCAL_MAP_SIZE--", 2048u);
	
	oclbackend.createProgram(kernels);
	
	return 0;
}
