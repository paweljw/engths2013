#include <iostream>
#include <string>
#include <fstream>

#define __SOLVERDEBUG
#define __SOLVERTIMING

#include "gpusolver.h"

using namespace std;

int main(int argc, char* argv[])
{
	string mtx_main = argv[1];
	string mtx_rhs = argv[2];

	fstream main;
	main.open(mtx_main.c_str(), std::fstream::in);

	if(!main.is_open()) 
	{
		cout << "Can't open " << mtx_main << endl;
		exit(1);
	}

	fstream rhs;
	rhs.open(mtx_rhs.c_str(), fstream::in);

	if(!rhs.is_open())
	{
		cout << "Can't open " << mtx_rhs << endl;
		exit(2);
	}

	string line;

	bool checked_size = false;
	int N, M, DP;

	cout << "MTX: Loading matrix size" << endl;

	while(getline(main, line))
	{
		if(line[0] != '%')
		{
			istringstream iss(line);

			if(!checked_size)
			{
				iss >> N >> M >> DP;
				cout << "N: " << N << ", M: " << M << ", DP: " << DP << endl;
				break;
			}
		}
	}

	PJWFront::GPUFrontal<float> gpuf(N);

	cout << "MTX: Loading matrix data" << endl;

	while(getline(main, line))
	{
		if(line[0] != '%')
		{
			istringstream iss(line);

			int x, y;
			float val;
			iss >> x >> y >> val;

			gpuf.set(x-1, y-1, val);
		}
	}

	checked_size = false;

	cout << "MTX: Loading RHS data" << endl;

	int counter = 0;
	while(getline(rhs, line))
	{
		if(line[0] != '%')
		{
			if(!checked_size) checked_size = true;
			else {
			istringstream iss(line);
			float val;
			iss >> val;

			gpuf.setRHS(counter, val);
			counter++;
			}
		}
	}

	cout << "Passing control to solver" << endl;

	gpuf.solve();

	cout << "Solver returned control" << endl;

	return 0;
}
