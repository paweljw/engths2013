#define __SOLVERDEBUG
#define __SOLVERTIMING
#include "gpusolver.h"
#include <iostream>

int main()
{
		PJWFront::GPUFrontal<float> vf(3, 1, 3);
		
		
		//vf.set(i, j, i+1);
		
		vf.set(0, 0, 1);
		vf.set(0, 1, 1);
		vf.set(0, 2, 1);
		vf.setRHS(0, 0);
		
		vf.set(1, 0, 1);
		vf.set(1, 1, -2);
		vf.set(1, 2, 2);
		vf.setRHS(1, 4);
		
		vf.set(2, 0, 1);
		vf.set(2, 1, 2);
		vf.set(2, 2, -1);
		vf.setRHS(2, 2);
		
		vf.solve();
		
		std::vector<float> solution = vf.solution;
		
		for(int i=0; i<3; i++)
			cout << solution[i] << " ";
			
		cout << endl;
		
		return 0;
}
