#ifndef kernels
#define kernels

/*
~~~~~Inputs~~~~~

A:array input to the program
N: where NxN is the problem
gen: generations to be played
cellsperthread: how many cells are assigned to one CUDA thread
*/

/*
~~~~~~Output~~~~~~
time elapsed (in seconds)
*/


/*Serial implementation of Game of Life.Each cell of the table is processed in serial 
and the output is written into another array,which is the input for the next generation 
(if there is any)*/
double life_0(int *A,int N,int gen);




/*Basic parallel implementation using 1 cell per 1 CUDA thread.The output is written into
another array per thread.After thread synchronization,the output array is given as input to the 
kernel for the next generation (if there is any)*/
double life_1(int *A,int N,int gen);



/*Parallel implementation using multiple cells per CUDA thread.A thread is calulating
next cell's condition for multiple cells.*/ 
double life_2(int *A,int N,int gen,int cellsperthread);



/*Parallel implementation using multiple cells per CUDA thread and shared memory.Each thread copies 
his "up" and "down" neighbors and his current onto shared memory.After synchronization,each cell 
computes its neighbors as above*/ 
double life_3(int *A,int N,int gen,int cellsperthread); //





#endif
