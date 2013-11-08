#include "oclbackend.h"
#include "shortmatrix.h"

int main()
{

	PJWFront::ocl::OCLBackend* backend = new PJWFront::ocl::OCLBackend();

	PJWFront::util::short_matrix<float> sm(4, backend);

	sm(0,0) = 1;
	sm(1,1) = 2;
	sm(2,2) = 3;
	sm(3,3) = 4;

	// assert number of data points
	unsigned int data_points = sm.__datapoints();
	assert(data_points == 10);

	// assert that data is ok
	float* data = sm.__flat_matrix();

	assert(data[0] == 1);
	assert(data[4] == 2);
	assert(data[7] == 3);
	assert(data[9] == 4);

	// assert proper row descriptors

	uint *ro = sm.__rowoffsets();

	assert(ro[0] == 0);
	assert(ro[1] == 4);
	assert(ro[2] == 7);
	assert(ro[3] == 9);
	
	// assert proper row function descriptors

	uint *rb = sm.__rowbegins();

	assert(rb[0] == 0);
	assert(rb[1] == 1);
	assert(rb[2] == 2);
	assert(rb[3] == 3);

	// all clear
	return 0;
/*
1 0 0 0
0 2 0 0
0 0 3 0
0 0 0 4

1 0 0 0 2 0 0 3 0 4

DP = 10
*/

}
