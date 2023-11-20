#include "PreprocessEngine.h"

int main() {
	int subject = 28;
	SSVEP data;
	PreprocessEngine* pe;

	data.loadCsv("./data/S028.csv");
	
	pe = new PreprocessEngine(&data);
	
	pe->notch(&data);
	//pe.filterBank(&data);


	return 0;
}

