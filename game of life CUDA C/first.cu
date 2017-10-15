#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

/*CUDA kernel 1*/
/*1 cell per thread*/
__global__ void kernel_1(int *input,int *output,int N)
{
	long int id=blockIdx.x*blockDim.x + threadIdx.x;//ID of each thread
	if(id<N*N)
	{	//2D mapping
		int i=(int)id/N;
		int j=id%N;

		int counter=0;

		//Neigbors
		int left=(!j)?N-1:j-1;
		int right=(j==N-1)?0:j+1;
		int up=(!i)?N-1:i-1;
		int down=(i==N-1)?0:i+1;


		counter=input[up*N+left]+input[up*N+j]+input[up*N+right]//top neighbors
		+input[i*N+left]+input[i*N+right]//same line neighbors
		+input[down*N+left]+input[down*N+j]+input[down*N+right];//bottom neighbors

		//next genetation
		output[id]=(counter==3)||((counter==2)&&input[id]);
	}
	
}


double life_1(int *A,int N,int gen)
{
	long int size=N*N;
	struct timeval stop,start;
	gettimeofday(&start, NULL);
	/* GPU */
	int *dev_A,*dev_B;//pointers to GPU
	/*Allocation for input and output arrays*/
	cudaMalloc(&dev_A,size*sizeof(int));
	cudaMalloc(&dev_B,size*sizeof(int));
	/*Initial state of game coped to input array to GPU*/
	cudaMemcpy(dev_A,A,size*sizeof(int),cudaMemcpyHostToDevice);
	/* GPU END */

	int thrB=512;//Number of threads per block
	int numB=(int)ceil(size/(float)thrB);//Number of blocks

	
	int i;
	
	int *input,*output,*temp;
	input=dev_A;
	output=dev_B;
	
	
	for(i=0;i<gen;i++)
	{
		/*Kernel call*/
		kernel_1<<<numB,thrB>>>(input,output,N);
		/*Thread synchronization*/
		cudaThreadSynchronize();

		temp=output;
		output=input;
		input=temp;
	}
	
	int *B=(int*)malloc(size*sizeof(int));//Output array
	/*result written for GPU to B array*/
	cudaMemcpy(B,temp,size*sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(dev_A);
	cudaFree(dev_B);
	free(B);

	gettimeofday(&stop, NULL);
	
	
	
	double tim=(double)((stop.tv_usec- start.tv_usec)/1.0e6 + stop.tv_sec -start.tv_sec);


	FILE *fp1;
	char filenameout[40];
	sprintf(filenameout,"table%dx%dout%dkernel1.bin",N,N,gen);
	fp1=fopen(filenameout,"w+");
	fwrite(B,sizeof(int),size,fp1);
	fclose(fp1);

	

	return tim;
	
}
