#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>
#include <string.h>

struct info
{
	int start,end,N,id,nt,sinks;
	int **conn;
	int *counter_in,*counter_out,*sink_node;
};

double *P_new,*P;
pthread_barrier_t barr1,barr2,barr3,barr4,barr5;
pthread_mutex_t lock;

int c;//itterations counter
int flag;//loop condition variable
double correction;//normalization factor
int *flags;

void *workers(void *arg);//declaration of threads' function

double parallel(int **conn,int *counter_in,int *counter_out,int N,int numb_th,double **result_p)
{
	struct timeval stop,start;
	gettimeofday(&start, NULL);

	int nod_thr=(int)N/numb_th;//number of nodes given to each thread
	int extras=N%numb_th;//extra nodes
	
	int i;
	flag=1;
	c=0;
	correction=0;


	flags=(int*)malloc(numb_th*sizeof(int));//loop condition for each thread

	pthread_t *thread=(pthread_t *)malloc(numb_th*sizeof(pthread_t));
	
	struct info *infos=(struct info *)malloc(numb_th*sizeof(struct info));

	P_new=(double*)malloc(N*sizeof(double));

	P=(double*)malloc(N*sizeof(double));

	pthread_barrier_init(&barr1, NULL, numb_th);
	pthread_barrier_init(&barr2, NULL, numb_th);
	pthread_barrier_init(&barr3, NULL, numb_th);
	pthread_barrier_init(&barr4, NULL, numb_th);
	pthread_barrier_init(&barr5, NULL, numb_th);

	pthread_mutex_init(&lock,NULL);
	
	for(i=0;i<numb_th;i++)//Initializing variables for each thread 
	{
		infos[i].start=(!i)?0:infos[i-1].end;
		infos[i].end=infos[i].start + nod_thr + (int)(extras>0);
		
		infos[i].conn=conn;
		infos[i].counter_in=counter_in;
		infos[i].counter_out=counter_out;
		
		infos[i].N=N;
		infos[i].id=i;
		infos[i].nt=numb_th;
		extras--;

		pthread_create(&thread[i],NULL,workers,(void*) &infos[i]);
	}
	
	for(i=0;i<numb_th;i++)
	{
		pthread_join(thread[i],NULL);
	}
	

	gettimeofday(&stop, NULL);
	double time_elapsed=(double)((stop.tv_usec- start.tv_usec)/1.0e6 + stop.tv_sec -start.tv_sec);

	printf("Iterations : %d\n",c );
	memcpy(*result_p,P,N*sizeof(double));
	
	free(P);
	free(P_new);
	free(thread);
	free(infos);
	free(flags);

	return time_elapsed;
}

void *workers(void *arg)
{
	
	struct info *infos=(struct info *)arg;
	
	int start=infos->start;
	int end=infos->end;
	int **conn=infos->conn;
	int *counter_in=infos->counter_in;
	int *counter_out=infos->counter_out;

	int N=infos->N;
	int id=infos->id;
	int nt=infos->nt;
	int i,j,k,l;
	
	for(i=start;i<end;i++)
	{
		P[i]=(double)1/N;
	}
	
	pthread_barrier_wait(&barr1);

	double d=0.85,tempc;
	double *temp;

	int lock_flag=1;
	while(flag)
	{
		pthread_barrier_wait(&barr2);
		
		lock_flag=0;
		flag=0;
		tempc=0;
		for(k=start;k<end;k++) {P_new[k]=0;}

		for(j=start;j<end;j++)
		{
			for(l=0;l<counter_in[j];l++)
			{
				i=conn[j][l];//ID of connected nodes to j
				P_new[j] += P[i]/counter_out[i];
			}
			if(!counter_out[j]) {tempc += P[j]/N;}//if node j is a sink node,treat him as he is connected to all the other nodes and himself
			//tempc is a correction variable tha normalizes the algorithm if there are sink nodes in the graph
			
		}
		pthread_mutex_lock(&lock);
		correction += tempc;//global correction
		pthread_mutex_unlock(&lock);

		pthread_barrier_wait(&barr3);

		for(j=start;j<end;j++)
		{
			P_new[j] = (P_new[j]+correction)*d + (1-d)/N;
			lock_flag |= fabs(P_new[j] - P[j])>0.0001;//Loop condition
		}
		flags[id]=lock_flag;

		pthread_barrier_wait(&barr4);

		if(id==0)//Thread 0 is the master thread
		{
			temp=P_new;//Pointer swap
			P_new=P;
			P=temp;
			flag=0;
			c++;
			correction=0;
			for(i=0;i<nt;i++)
			{
				flag |= flags[i];
			}
		}
		pthread_barrier_wait(&barr5);
	}
	pthread_exit(NULL);
}
