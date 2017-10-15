#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "functions.h"
#include <pthread.h>
#include <limits.h>
#include <errno.h>

int check(double *a,double *b,int N); //function that checks the elements of 2 arrays

int main(int argc,char **argv)
{
	
	if(argc != 2)
	{
		printf("Usage: %s filename\n"
			"where \n"
			"filename: name of input file\n",argv[0]);
		return 1;

	}
	
	/*File reading */
	FILE *fp=fopen(argv[1],"r");

	if(fp==NULL)
	{
		char str[50];
		sprintf(str,"Error, '%s' ",argv[1]);
		perror(str);
		return 2;
	}


	int i,N;

	/*Reading from file */
	int a,b;
	int min=INT_MAX;
	int max=INT_MIN;
	
	while(fscanf(fp,"%d %d\n",&a,&b)!=EOF)//finding the max and min IDs of the graph
	{
		if(a>max) {max=a;}
		if(b>max) {max=b;}
		if(a<min) {min=a;}
		if(b<min) {min=b;}
	}

	N=max-min+1; //Number of nodes
	
	if(N<0) { printf("Error in file reading\n");return 3; }

	//Allocating arrays
	int **conn=(int **)malloc(N*sizeof(int *));//conn[][] stores for each node,which nodes are connected to him
	int *counter_out=(int*)calloc(N,sizeof(int));//counter_out[] stores the number of outgoing nodes from each node
	int *counter_in=(int*)calloc(N,sizeof(int));//counter_in[] stores the number of incoming nodes to each node
	

	rewind(fp);//Go to the start of the file

	while(fscanf(fp,"%d %d\n",&a,&b)!=EOF)
	{
		a=a-min;//So the lower ID of a node be 0
		b=b-min;
		conn[b]=(int *)realloc(conn[b],(counter_in[b]+1)*sizeof(int));
		conn[b][counter_in[b]]=a;//a is an incoming node to b
		counter_in[b]++;
		counter_out[a]++;
	}

	fclose(fp);

	/* All arrays are ready */
	double time_serial,time_parallel;
	double *result_s,*result_p;

	result_s=(double *)malloc(N*sizeof(double));
	result_p=(double *)malloc(N*sizeof(double));

	printf("-----------START-----------\n\n");
	printf("Pagerank for %d nodes\n",N );
	printf("Serial algorithm\n");
	time_serial=serial(conn,counter_in,counter_out,N,&result_s);
	printf("Computation time in seconds : %f\n",time_serial);
	int numb_th;
	for(numb_th=2;numb_th<128;numb_th *= 2)
	{
			printf("\nParallel algorithm using %d threads\n",numb_th );
			time_parallel=parallel(conn,counter_in,counter_out,N,numb_th,&result_p);
			printf("Computation time in seconds : %f\n",time_parallel);
			
			printf("\nComparison status : ");
			if(check(result_p,result_s,N)) {printf("Success\n");}
			else {printf("Fail\n");}
	}

	free(counter_in);
	free(counter_out);
	for(i=0;i<N;i++) {free(conn[i]);}
	free(conn);
	free(result_s);
	free(result_p);

	return EXIT_SUCCESS;
}


int check(double *a,double *b,int N)
{
	int i,c=1;
	for(i=0;i<N;i++)
	{
		c &= fabs(a[i]-b[i])<=0.000000001;
	}
	return c;

}
