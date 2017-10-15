#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

struct timeval startwtime, endwtime;
double par_time,ser_time;

const int up=1;
const int down=0;
const int tag=0;
int *a,N,*for_sort,*for_serial,rank,np,last_rec=0,*buffer,group=2;


/*Functions used in serial recursive bitonic sort */
inline void ser_exchange(int i,int j);
void recBitonicSort(int lo,int cnt,int dir);
void ser_compare(int i,int j,int dir);
void ser_sort(void);
void ser_bitonicMerge(int lo, int cnt, int dir);
/*Functions used in parallel recursive bitonic sort*/
int data_transfer(int spots,int dir);
void collect_data(void);
void par_sort(void);
inline int compareup(const void * a, const void * b);
inline int comparedown(const void * a, const void * b);
/*General Funtions*/
int test(int *temp,int x);
void print(int *temp,int N);
void init(void);

/*~~~~~~~~~~~~~~~~~~ Main Function ~~~~~~~~~~~~~~~*/

int main(int argc,char **argv)
{
	if(argc != 3)
	{
		printf("Usage: %s q p\n where N=2^q is problem size per task (power of two)\n and np=2^p tasks\n",argv[0]);
		exit(1);
	}
	N=1<<atoi(argv[1]);
	np=1<<atoi(argv[2]);
	if((a=(int *)malloc(N*sizeof(int)))==NULL)
  {
    printf("Error allocating memory\n");
    exit(2);
  }
  if(!rank)//Only serial process
  {
    if((for_serial=(int *)malloc(np*N*sizeof(int)))==NULL)
    {
      printf("Error allocating memory\n");
      exit(2);
    }
  }
  if(MPI_Init(&argc,&argv)!=MPI_SUCCESS)
  {
    printf("Error Initiallazing\n");
    exit(3);
  }
  for_sort=(int *)malloc(np*N*sizeof(int));
	MPI_Comm_size(MPI_COMM_WORLD,&np);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	init();//Array initialization of each process
  MPI_Barrier(MPI_COMM_WORLD);//Barrier for the initialization
  ser_sort();
  MPI_Barrier(MPI_COMM_WORLD);
  print(a,N);
  gettimeofday (&startwtime, NULL);
	par_sort();
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday (&endwtime, NULL);
  par_time=(double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);
  collect_data();
  if(!rank)
  {
    printf("Faster : %f\n",ser_time/par_time );
  }
  MPI_Finalize();
  free(for_serial);
  free(for_sort);
  return 0;
}

/*Functions that used*/
/*Test for sorted array*/
int test(int *temp,int x) {
  int pass = 1;
  int i;
  for (i = 1; i < x; i++) {
    pass &= (temp[i-1] <= temp[i]);
  }
  //printf(" TEST %s\n",(pass) ? "PASSed" : "FAILed");
  return pass;
}
/*Initialization of local array*/
void init() {
  int i;
  srand(time(NULL)+rank);
  for (i = 0; i < N; i++) {
    a[i] = rand() % N; // (N - i);
  }
  /*int i=0;
  for(i;i<N;i++)
  {
    a[i]=N-i;
  }*/
}
/*Array printing function */
void print(int *temp,int x) {
  int i;
  for (i = 0; i < x; i++) {
    printf("%d--", temp[i]);
    if(!((i+1)%N)) printf("|||\n");
  }
  printf("\n");
}
/*~~~~~~~~~~~~~~~~~~~~~~~~~~ For parallel functions ~~~~~~~~~~~~~~~~~~~~~~~~*/
/*Comparing function for ascending sort*/
inline int compareup(const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}

inline int comparedown(const void * a, const void * b)
{
  return ( *(int*)b - *(int*)a );
}
/*Exchanging buffer with local array*/
inline void exchange_test(int i,int start)
{
  int temp;
  temp=a[i+start];
  a[i+start]=buffer[i];
  buffer[i]=temp;
}
/*Exchanging elements, depending on what half of array we are*/
void compare_test(int x,int dir)
{
  int i;
  for(i=0;i<N/2;i++)
  {
    if(!x){
      if(dir==a[i]>buffer[i]) exchange_test(i,0);
    }
    else
    {
      if(dir==a[i+N/2]<=buffer[i]) exchange_test(i,N/2);
    }
  }
}
//Printing
void printwhole(int dist)
{
  for_sort=(int*)malloc(N*sizeof(int)*np);
  MPI_Gather(a,N,MPI_INT,for_sort,N,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if(!rank){
    printf("%d\n",dist );
    print(for_sort,N*np);
  } 

}

int data_transfer(int spots,int dir)
{
  if(spots>=2)
  {
    int dist=(int)log2(spots)-1;
    int handshake=rank^(1<<dist);
    MPI_Request req;
    MPI_Status stat;
    buffer = (int*)calloc((N/2),sizeof(int));
    int *buffer2= (int*)calloc((N/2),sizeof(int));
    int i;
    int start=rank<handshake?N/2:0;
    for(i=0;i<N/2;i++)
    {
      buffer2[i]=a[i+start];
    }
    /*Non-blocking couple element sending and blocking receiving*/
    MPI_Isend(buffer2,N/2,MPI_INT,handshake,tag,MPI_COMM_WORLD,&req);
    MPI_Recv(buffer,N/2,MPI_INT,handshake,tag,MPI_COMM_WORLD,&stat);
    compare_test((rank>=handshake),dir);
    MPI_Wait(&req,&stat);//block until exchange is over
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Isend(buffer,N/2,MPI_INT,handshake,tag,MPI_COMM_WORLD,&req);
    MPI_Recv(buffer2,N/2,MPI_INT,handshake,tag,MPI_COMM_WORLD,&stat);
    for(i=0;i<(N/2);i++) a[i+start]=buffer2[i];
    MPI_Wait(&req,&stat);
    if(last_rec)
    {
  
      if(spots==2) qsort(a,N,sizeof(int),compareup);
      data_transfer(spots/2,up);
      return 0;
    }
    int bit=(!(rank&(1<<((int)log2(group)))));
    if(spots==2)
    {

      if(bit)
      {
        qsort(a,N,sizeof(int),compareup);
      }
      else 
      {
        qsort(a,N,sizeof(int),comparedown);
      }
    }
    //printwhole(dist);
    data_transfer(spots/2,bit);
  }
}

void collect_data()
{
  for_sort=(int*)malloc(3*np*sizeof(int)); 
  int temp[3];
  temp[0]=a[0];
  temp[1]=a[N-1];
  temp[2]=test(a,N);

  //MPI_Gather(a,N,MPI_INT,for_sort,N,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Gather(temp,3,MPI_INT,for_sort,3,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if(!rank)
  {
    //test(for_sort,N*np);//ελεγχος
    //print(for_sort,N*np);
    int j=1;

    int pass=for_sort[2];
    for(j;j<np;j++)
    {
      pass&=((for_sort[3*j-2]<=for_sort[3*j])&(for_sort[3*j - 1 ]));
      if(!pass) break;
    }
    printf("My_Test %s ed \n",(pass)?"PASS":"FAIL" );
    printf("Parallel wall clock time : %f\n",par_time);
  }
}

void par_sort(void)
{
	//Bitonic ανα 2 διπλανα μηχανηματα
  if(rank%2)
  {
    qsort(a,N,sizeof(int),comparedown);
  }
  else
  {
    qsort(a,N,sizeof(int),compareup);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  while (group<=np)
  {
    last_rec=(group==np);
    int d=(int)log2(group);
    int dir=(!(rank&(1<<d)));
    data_transfer(group,(last_rec)?up:dir);
    group=group*2;
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Serial Code~~~~~~~~~~~~~~~~~~~~~~*/


void ser_sort(void)
{
  //Μαζευουμε τους πινακες απο καθε process για να τρεξουμε το ιδιο προβλημα
  MPI_Gather(a,N,MPI_INT,for_serial,N,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday (&startwtime, NULL);
  if(!rank)
  recBitonicSort(0, N*np, up);//Μονο το process 0 τρεχει τον σειριακο κωδικα
  gettimeofday (&endwtime, NULL);
  ser_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
          + endwtime.tv_sec - startwtime.tv_sec);
  /*if(!rank){
    //test(for_serial,N*np);//Ελεγχος
    //printf("Recursive wall clock time = %f\n", ser_time);
  }*/
}


void ser_bitonicMerge(int lo, int cnt, int dir) {
  if (cnt>1) {
    int k=cnt/2;
    int i;
    for (i=lo; i<lo+k; i++)
      ser_compare(i, i+k, dir);
    ser_bitonicMerge(lo, k, dir);
    ser_bitonicMerge(lo+k, k, dir);
  }
}

void recBitonicSort(int lo, int cnt, int dir) {
  if (cnt>1) {
    int k=cnt/2;
    recBitonicSort(lo, k,up);
    recBitonicSort(lo+k, k, down);
    ser_bitonicMerge(lo, cnt, dir);
  }
}

void ser_compare(int i,int j,int dir)
{
  if (dir==(for_serial[i]>for_serial[j])) 
    ser_exchange(i,j);
}

inline void ser_exchange(int i,int j)
{
  int t;
  t = for_serial[i];
  for_serial[i] = for_serial[j];
  for_serial[j] = t;
}




