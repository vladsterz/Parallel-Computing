//-----------------Functions definitons------------------
#ifndef functions
#define functions

/* **conn (conn[i][j])             : This array holds for node i , the IDs of incoming nodes
	*counter_in (counter_in[i])    : This array holds the number of incoming nodes to node i
	*counter_out (counter_out[i])  : This array holds the number of outgoing nodes from node i
	N                              : Number of nodes
	**result  (*result[i])         : The final array containing the odds for each node
	numb_th                        : Number of threads to use in the parallel implementation */
double serial(int **conn,int *counter_in,int *counter_out,int N,double **result_s);

double parallel(int **conn,int *counter_in,int *counter_out,int N,int numb_th,double **result_p);

/*	return value                   : Time elapsed since the call of the function */

#endif 
