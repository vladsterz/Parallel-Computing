#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>


double serial(int **conn,int *counter_in,int *counter_out,int N,double **result_s)
{
	struct timeval stop,start;
	gettimeofday(&start, NULL);

	double *P,*P_new,*temp;
	P=(double *)calloc(N,sizeof(double));
	P_new=(double *)malloc(N*sizeof(double));
	
	double d=0.85,tempc;
	int i,j,k,l,c;
	
	for(i=0;i<N;i++) {P[i]=(double)1/N;}//Initiallazing P
	int flag=1;//While condition
	c=0;//itterations counter
	while (flag)
	{
		flag=0;
		tempc=0;

		for(k=0;k<N;k++) {P_new[k]=0;}
		
		for(j=0;j<N;j++)
		{
			for(l=0;l<counter_in[j];l++)
			{
				i=conn[j][l];//ID of connected nodes to j
				P_new[j] += P[i]/counter_out[i];
			}
			if(!counter_out[j]) {tempc += P[j]/N;}//if node j is a sink node,treat him as he is connected to all the other nodes and himself
			//tempc is a correction variable tha normalizes the algorithm if there are sink nodes in the graph
		}
		for(j=0;j<N;j++)
		{
			P_new[j] = (P_new[j]+tempc)*d + (1-d)/N;
			flag |= fabs(P_new[j] - P[j])>0.0001;//Loop condition
		}
		//Pointer swapping to so P=P_new
		temp=P_new;
		P_new=P;
		P=temp;
		c++;
	}
	gettimeofday(&stop, NULL);
	double time_elapsed=(double)((stop.tv_usec- start.tv_usec)/1.0e6 + stop.tv_sec -start.tv_sec);
	printf("Iterations : %d\n",c);
	memcpy(*result_s,P,N*sizeof(double));

	free(P);
	free(P_new);

	return time_elapsed;
}
