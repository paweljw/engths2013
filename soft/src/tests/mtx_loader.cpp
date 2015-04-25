#include <boost/lexical_cast.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#define __SOLVERDEBUG
//#define __SOLVERTIMING
#define __SOLVERTIMING_SILENT
#define __SOLVER_DOT

#include "gpusolver.h"

using namespace std;

int main(int argc, char* argv[])
{
	// cout << "argc is " << argc << endl;
	string mtx_main = argv[1];
	string mtx_rhs = argv[2];

	string _platform = argv[5];
	string _device = argv[6];
	string _offset = argv[7];

	int platform = boost::lexical_cast<int>(_platform);
	int device = boost::lexical_cast<int>(_device);
	int mtx_offset = boost::lexical_cast<uint>(_offset);

	unsigned int _LWS = 192;
	unsigned int _GWS = 0;

	string impl;

	bool fakeRHS = false;

	if(argc > 3)
	{
		string sLWS = argv[3];
		string sGWS = argv[4];

		istringstream isLWS(sLWS);
		istringstream isGWS(sGWS);

		isLWS >> _LWS;
		isGWS >> _GWS;
	}

	if(argc > 5)
	{
		impl = argv[5];
	}

	//cout << "Opening files..." << endl;

	fstream main;
	main.open(mtx_main.c_str(), std::fstream::in);

	if(!main.is_open())
	{
		cout << "Can't open " << mtx_main << endl;
		exit(1);
	}

	if(mtx_rhs == "FAKE") fakeRHS = true;

	fstream rhs;

	if(!fakeRHS)
	{
		rhs.open(mtx_rhs.c_str(), fstream::in);

		if(!rhs.is_open())
		{
			cout << "Can't open " << mtx_rhs << endl;
			exit(2);
		}
	}

	//cout << "Opened dem files" << endl;

	string line;

	bool checked_size = false;
	int N, M, DP;

	while(getline(main, line))
	{
		if(line[0] != '%')
		{
			istringstream iss(line);

			if(!checked_size)
			{
				iss >> N >> M >> DP;
				break;
			}
		}
	}

//cout << "Initializing solver " << N << endl;

	PJWFront::GPUFrontal<double> gpuf(N, _LWS, _GWS, platform, device, mtx_offset);

//cout << "Initialized solver" << endl;
/*

	cout << "MTX: " << mtx_main << endl;
	cout << "RHS: " << mtx_rhs << endl;
	cout << "GWS: " << gpuf.GWS << endl;
	cout << "LWS: " << gpuf.LWS << endl;
	cout << "N:   " << N << endl << "M:   " << M << endl << "DP:   " << DP << endl;

	cout << "Upycham macierz" << endl;
*/
	while(getline(main, line))
	{
		if(line[0] != '%')
		{
			istringstream iss(line);

			int x, y;
			float val;
			iss >> x >> y >> val;

			gpuf.set(x, y, val);
		}
	}

	checked_size = false;

//	cout << "Upycham RHS" << endl;

	if(!fakeRHS)
	{
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
	} else {
		for(int i=0; i<N;i++) gpuf.setRHS(i, 1);
	}

//	cout << "Passing control to solver" << endl;

	try
	{
//		cout << "In try" << endl;
		gpuf.solve();
	} catch(PJWFront::BadException) {
		cout << "General bad exception happened" << endl;
		return 0;
	} catch(PJWFront::UnsolvableException) {
		cout << "System appears to be unsolvable" << endl;
		//return 0;
	} catch(PJWFront::NanException) {
		cout << "The NaN error appeared" << endl;
		return 0;
	}

cout << "----";

	printf("\nTIME: %0.5f s\n", gpuf.ocltime);
	printf("FLOPS: %0.3f\n", gpuf.fmads);


	return 0;
}
