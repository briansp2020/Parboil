#ifndef _KERNEL_H_
#define _KERNEL_H_

typedef int2 Node;
typedef int2 Edge;

extern texture<Node, 1, hipReadModeElementType> g_graph_node_ref;
extern texture<Edge, 1, hipReadModeElementType> g_graph_edge_ref;

__global__ void
BFS_kernel(grid_launch_parm lp,
           int * q1, 
           int * q2, 
           Node* g_graph_nodes, 
           Edge* g_graph_edges, 
           int* g_color, 
           int * g_cost, 
           int no_of_nodes, 
           int * tail, 
           int gray_shade, 
           int k);

#endif
