#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

double life_0(int *A,int N,int gen)
{
	struct timeval stop,start;
	gettimeofday(&start, NULL);
	
	/*Input and output arrays*/
	int long size=N*N;
	int *output=(int*)malloc(size*sizeof(int));
	int *input=(int*)malloc(size*sizeof(int));
	memcpy(input,A,size*sizeof(int));
	int *temp;//temporary pointer to swap input and output

	int c=0;
	
	int i,j,counter;
	int left,right,up,down;//Neigbors of each cell
	for(c=0;c<gen;c++)//Generation loop
	{
		for(i=0;i<N;i++)
		{
			//up and down neighbors
			up=(!i)?N-1:i-1;
			down=(i==N-1)?0:i+1;
			for(j=0;j<N;j++)
			{
				counter=0;
				//left and right neigbors
				left=(!j)?N-1:j-1;
				right=(j==N-1)?0:j+1;
				

				counter=input[up*N+left]+input[up*N+j]+input[up*N+right]//up neigbors
				+input[i*N+left]+input[i*N+right] //left and right neighbors
				+input[down*N+left]+input[down*N+j]+input[down*N+right];//down neighbors
				//next generation 			
				output[i*N+j]=(counter==3)||((counter==2)&&input[i*N+j]);
			}
		}
			
			

		//pointer swap
		temp=output;
		output=input;
		input=temp;

	}


	FILE *fp1;
	char filenameout[40];
	sprintf(filenameout,"table%dx%dout%dkernel0.bin",N,N,gen);
	fp1=fopen(filenameout,"w+");
	fwrite(temp,sizeof(int),size,fp1);
	fclose(fp1);
	free(output);
	free(input);


	gettimeofday(&stop, NULL);
	double tim=(double)((stop.tv_usec- start.tv_usec)/1.0e6 + stop.tv_sec -start.tv_sec);
	return tim;
}
