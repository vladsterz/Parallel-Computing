#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

/*CUDA kernel 2*/
/*Multiple cells per thread*/
__global__ void kernel_2(int *input,int *output,int N,int cellsperthread)
{
	//Thread ID
	int t=(int)log2((float)cellsperthread);//Number of shifts to do
	long int id=blockIdx.x*blockDim.x + threadIdx.x;
	
	id=id<<t;
	//id=id*cellsperthread;
	
	//2D mapping
	int i=(int)id/N;
	int j=id%N;
	if(id<(N*N))
	{
		
		int counter;
		int left,right,up,down;
		int tt=0;
		int lu,ld,su,sd;
		
		for(tt=0;tt<cellsperthread;tt++,j++)
		{
			counter=0;
			//Neigbors
			right=(j==N-1)?0:j+1;
			up=(!i)?N-1:i-1;
			down=(i==N-1)?0:i+1;
			if((!tt)||(!j))//First cell has to do extra work
			{
				lu=input[up*N + j];//upper left neighbor for the NEXT cell
				ld=input[down*N + j];//down left neighbor for the NEXT cell
				su=input[up*N + right];//up neigbor for the NEXT cell
				sd=input[down*N + right];//down neigbor for the NEXT cell

				left=(!j)?N-1:j-1;
				//Left neighbors won't be needed for the next cell
				counter = input[up*N+left]+input[i*N+left]+input[down*N+left];
				counter+= lu + ld + su + sd + input[i*N + right];
			}
			else
			{
				counter = lu + ld + su + sd + input[i*N+ j-1];//values from previous cell
				//passing arguments for the next cell
				lu=su;
				ld=sd;
				su=input[up*N + right];
				sd=input[down*N +right];
				counter += su + sd + input[i*N + right];
			}
			output[i*N+j]=(counter==3)||((counter==2)&&input[i*N+j]);//emit output

			if(j==N-1) 
			{
				if(i==N-1) break;
				i++;
				j=-1;
			}


		}
	}
}


double life_2(int *A,int N,int gen,int cellsperthread)
{
	struct timeval stop,start;
	gettimeofday(&start, NULL);
	

	int thrB=512;//Threads per block
	long int size=N*N;
	int numB=(int)ceil(size/(float)(thrB*cellsperthread));//Number of blocks

	/* GPU */
	int *dev_A,*dev_B;
	cudaMalloc(&dev_A,size*sizeof(int));
	cudaMalloc(&dev_B,size*sizeof(int));
	cudaMemcpy(dev_A,A,size*sizeof(int),cudaMemcpyHostToDevice);
	/*GPU*/

	int i=0,*input,*output,*temp;
	input=dev_A;
	output=dev_B;
	
	for(i=0;i<gen;i++)
	{
		/*Kernel call*/
		kernel_2<<<numB,thrB>>>(input,output,N,cellsperthread);
		/*Thread Synchronization*/
		cudaThreadSynchronize();

		/*Pointer Swap*/
		temp=output;
		output=input;
		input=temp;
	}
	
	int *B=(int*)malloc(size*sizeof(int));
	/*Result*/
	cudaMemcpy(B,temp,size*sizeof(int),cudaMemcpyDeviceToHost);
	
	cudaFree(dev_A);
	cudaFree(dev_B);
	free(B);

	gettimeofday(&stop, NULL);
	double tim=(double)((stop.tv_usec- start.tv_usec)/1.0e6 + stop.tv_sec -start.tv_sec);

	FILE *fp1;
	char filenameout[40];
	sprintf(filenameout,"table%dx%dout%dkernel2.bin",N,N,gen);
	fp1=fopen(filenameout,"w+");
	fwrite(B,sizeof(int),size,fp1);
	fclose(fp1);

	
	
	return tim;
}

