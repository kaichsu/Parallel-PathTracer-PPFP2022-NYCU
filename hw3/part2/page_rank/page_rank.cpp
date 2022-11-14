#include "page_rank.h"
#include <vector>
#include <omp.h>
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <utility>
#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  double* old_sol = (double *)malloc(numNodes*sizeof(double));
  bool converged = false;
  double sink = 0;
  double diff = 0;
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
  }

  while(!converged){
    sink = 0;
    diff = 0;
    #pragma omp parallel for reduction(+:sink)
    for(int i=0; i<numNodes; ++i){
      old_sol[i] = solution[i];
      if(outgoing_size(g,i) == 0){
        sink += old_sol[i];
      }
    }
    sink *= damping / numNodes;
    
    #pragma omp parallel for reduction(+:diff)
    for(int i=0; i<numNodes; ++i){
      solution[i] = 0;
      const Vertex* start = incoming_begin(g, i);
      const Vertex* end = incoming_end(g, i);
      for(const Vertex* v=start; v!= end; ++v){
        solution[i] += old_sol[*v]/outgoing_size(g, *v);
      }
      solution[i] = damping * solution[i] + (1.0-damping) / numNodes;
      solution[i] += sink;
      diff += fabs(solution[i] - old_sol[i]);
    }
    if(diff <= convergence){
      converged = true;
      break;
    }
  }
  free(old_sol);
  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }
   */
    
}
