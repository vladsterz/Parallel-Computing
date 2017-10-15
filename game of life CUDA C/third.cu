#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>



__global__ void kernel_3(int *input,int *output,int N,int thrB,int cellsperthread)
{
	long int size=N*N;
	long int id=blockIdx.x*blockDim.x + threadIdx.x;//ID of thread
	int t=(int)log2((float)cellsperthread);//times to shift
	if( id <(size>>t))
	{
		//2D mapping
		int i=(int)id/N;
		int j=id%N;
		//row-wise work for threads
		i=i<<cellsperthread;
		int tt=0;
		__shared__ int local[3][500];//shared memory
		int pos=threadIdx.x;
		int su,ss,lu,ls,ru,rs;//values for NEXT cells
		int up,left,right,down;//neigbors
		int self;
		int flag=0;
		int counter=0;

		for(tt=0;(tt<cellsperthread)||flag;tt++,i++)
		{
			
			counter=0;
			//neigbors
			right=(j==N-1)?0:j+1;
			up=(!i)?N-1:i-1;
			down=(i==N-1)?0:i+1;
			left=(!j)?N-1:j-1;
			__syncthreads();
			if(!tt)
			{
				//UP values wont be needed for the next cells
				counter+= local[0][pos]=input[up*N+j];
				self = su=local[1][pos]=input[i*N+j];//up neighbor for the NEXT cell
			}
			else
			{
				counter+= local[0][pos]=su;//previous values
				self = su=local[1][pos]=ss;
			}
			counter+= ss=local[2][pos]=input[down*N+j];

			if(!pos)//Threads with local id=0 and thrB-1 have to extra work
			{

				if(!tt)//First row extra reads
				{
					counter+= input[up*N + left];
					counter+= lu=input[i*N + left];//upper left neighbor for NEXT cell
					counter+= ls=input[down*N + left];//left neighbor for NEXT cell
				}
				else//every other row reads once from global memory
				{
					counter+=lu+ls;
					lu=ls;
					counter+= ls=input[down*N + left];
				}
			}
			else if(pos==thrB-1||id== size -1) //thread thrB-1 extra work
			{
				//Same as above only with right neigbors
				if(!tt)
				{
					counter+= input[up*N + right];
					counter+= ru=input[i*N + right];
					counter+= rs=input[down*N + right];
				}
				else
				{
					counter+=ru+rs;
					ru=rs;
					counter+= rs=input[down*N + right];
				}
			}
			__syncthreads();
			int ii=0;
			for(ii=0;ii<3;ii++)
			{
				//threads with ID=0 and ID=thrB-1 have already compute left or right neighbors
				counter+=(pos!=0)?local[ii][pos-1]:0;
				counter+=(pos!=thrB-1)?local[ii][pos+1]:0;
			}
			output[i*N+j]=(counter==3)||((counter==2)&&self);
			flag=((N-i)<cellsperthread)?1:0;//chech if there is extra work to do for the last block of threads
			if(i==N-1) break;
		}

		
	}

}
double life_3(int *A,int N,int gen,int cellsperthread)
{
	
	struct timeval stop,start;
	gettimeofday(&start, NULL);


	long int size=N*N;
	int thrB=500;//Number of threads within a block
	int numB=(int)ceil(size/(float)(thrB*cellsperthread));//Number of blocks
	int *c,*input,*output,*temp;


	/*GPU*/
	int *dev_A,*dev_B;
	cudaMalloc(&dev_A,size*sizeof(int));
	cudaMalloc(&dev_B,size*sizeof(int));
	cudaMemcpy(dev_A,A,size*sizeof(int),cudaMemcpyHostToDevice);

	input=dev_A;
	output=dev_B;

	
	int i=0;
	for(i=0;i<gen;i++)
	{
		/*Kernel 3 launch*/	
		kernel_3<<<numB,thrB>>>(input,output,N,thrB,cellsperthread);
		/*Thread synchronization*/
		cudaThreadSynchronize();

		/*pointer swap*/
		temp=output;
		output=input;
		input=temp;
	}
	
	c=(int*)malloc(size*sizeof(int));
	cudaMemcpy(c,temp,size*sizeof(int),cudaMemcpyDeviceToHost);

	cudaFree(dev_A);
	cudaFree(dev_B);
	free(c);

	


	gettimeofday(&stop, NULL);
	double tim=(double)((stop.tv_usec- start.tv_usec)/1.0e6 + stop.tv_sec -start.tv_sec);


	FILE *fp1;
	char filenameout[40];
	sprintf(filenameout,"table%dx%dout%dkernel3.bin",N,N,gen);
	fp1=fopen(filenameout,"w+");
	fwrite(c,sizeof(int),size,fp1);
	fclose(fp1);
	
	return tim;

}
