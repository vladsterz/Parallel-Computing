#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include "kernels.h"

/*Vladimiros Sterzentsenko*/
/*Parallel and Distributed System 2015 work 3*/

/*Main function*/
int main(int argc,char **argv)
{

	if(argc!=5)
	{
		printf("Usage: %s filename,N,gen,cellsperthread,kernel\n"
		"\nfilename:Name of input file\n"
		"N:Number of rows of the input file\n"
		"gen:Generations to be played\n"
		"cellsperthread (2nd and 3d kernels):How many cells per thread(power of 2)\n", argv[0]);
			exit(1);
	}

	/*Inputs to the program*/
	char *filename=argv[1];
	int N=atoi(argv[2]);
	int gen=atoi(argv[3]);
	int cellsperthread=atoi(argv[4]);//power of 2
	//int kernel=atoi(argv[5]);
	
	FILE *fp=fopen(filename,"r+");
	int *A;
	if(fp!=NULL)
	{
		A=(int*)malloc(N*N*sizeof(int));//must free
		fread(A,sizeof(int),N*N,fp);
		fclose(fp);
	}
	else
	{
		printf("%s\n", strerror(errno));
		printf("ERROR OPENING FILE,EXITING\n");
		return 1;
	}
	
	//cellsperthread=4;
	
	for(gen=10;gen<1001;gen=gen*10){
	double ser_time=life_0(A,N,gen);
	cudaDeviceSynchronize();
	cudaDeviceReset();
	printf("N=%d,gen=%d,cells/thread=%d\n",N,gen,cellsperthread);
	printf("%f\n",ser_time);
	printf("%f\n",life_1(A,N,gen));
	cudaDeviceSynchronize();
	cudaDeviceReset();
	printf("%f\n",life_2(A,N,gen,cellsperthread));
	cudaDeviceSynchronize();
	cudaDeviceReset();
	printf("%f\n",life_3(A,N,gen,cellsperthread*2) );
}
	free(A);
	return 0;

}
