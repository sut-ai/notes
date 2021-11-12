
# Uninformed Search

Every problem around us should be modeled and formulated in the first step if we want to solve them. After that we must find a solution and solution is an action sequence, in order to do that we must "search". This is where search algorithm are being used and developed. The search possible action sequences starting at the initial state form a search tree with the initial state node at the root; the branches are actions and the nodes correspond to states in the state space of the problem. the very first thing we have to do is to check that if the current state is the goal node or not. If it is not we must take various actions and we do this by expansion; expandind the current state. This would lead to generating a new set of states from parent node and new states are called child nodes. Depends on the search strategy we chose, we proceed to reach out the goal state and then we stop.

# Contents 

[Introduction](#Introduction)

[Breadth-first Search](#Breadth-first-Search)

[Uniform cost search](#Uniform-cost-search)

[Depth-first Search](#Depth-first-Search)

[Depth-limited Search](#Depth-limited-Search)

[Iterative deepening depth-first search](#Iterative-deepening-depth-first-search)

[Bidirectional search](#Bidirectional-search)

[Conclusion](#Conclusion)

## Introduction
As the name suggests, Uninformed search and algorithms try to reach the goal state blindly which means they don't have and save extra and additional information about search space. They operate in a brute force way and use no domain knowlege for operating and they just search and move forward, whether it is a right way and route or not until they reach the goal state.

There are two kinds of search; Tree search and Graph search which is explained below.
In tree search, while using the main strategy, we consider expanded nodes again if it's neseccary or on the way. But in graph search we will not do that and the expanded nodes will be ignored and will not get expanded again. It is obvious that Graph search is much faster than tree search.

## Breadth-first Search
It is one of the algorithms to search a graph or a tree to find a specific thing we’re looking for. It is an uninformed kind of search and uses memory to search nodes. The procedure of it is that the algorithm starts from a node and explores all the nodes above it that are in the same level and it chooses the shallowest unexpanded node for expansion. This algorithm has some similarities with the DFS algorithm but it is different in some ways. 

![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/bfs.jpg)
 
Now let’s talk about this algorithm features and performance measures.

**Time complexity**

It is O(b^(d+1)). This is an exponential complexity which is so much

**Space complexity**

This algorithm keeps every nodes in its memory, thus it would be equal to O(b^(d+1))

**Completeness**

It’s a complete algorithm only and only that if b is finite.

**Optimality**

It is not an optimal algorithm in general but if cost of steps be equal to one, this algorithm would be optimal.

As it is obvious this algorithm has some pros and cons. The benefit of it is that it is accurate and easy to run and implement. But if the goal be at depth 21, this algorithm would take hundreds of years to find the solution.

As it is obvious from the example below, the algorithm searches aevery node in a level and then goes to next level .
![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/breadth-first-search.png)
## Uniform-Cost Search 

Uniform-Cost Search (**UCS**) is a variant of Dijikstra’s algorithm. Here, instead of inserting all vertices into a priority queue, we insert only source, then one by one insert when needed. In every step, we check if the item is already in priority queue (using visited array). If yes, we perform decrease key, else we insert it. 
This variant of Dijkstra is useful for infinite graphs and those graph which are too large to represent in the memory. Uniform-Cost Search is mainly used in Artificial Intelligence.

|Step|Description|Shape|
|----|-----------|-----|
|0|Example graph |![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/UCS_1.jpg)|
|1|First, we just have the source node in the queue |![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/UCS_2.jpg)|
|2|Then, we add its children to the priority queue with their cumulative distance as priority |![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/UCS_3.1.jpg)|
|3|Now, A has the minimum distance (i.e., maximum priority), so it is extracted from the list. Since A is not the destination, its children are added to the PQ.(in pervious step node 1 had minimun cost)|![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/UCS_3.jpg)|
|4|B has the maximum priority now, so its children are added to the queue|![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/UCS_4.jpg)|
|5|Up next, G will be removed and its children will be added to the queue|![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/UCS_5.jpg)|
|6|C and I have the same distance, so we will remove alphabetically|![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/UCS_6.jpg)|
|7|Next, we remove I; however, I has no further children, so there is no update to the queue. After that, we remove D. D only has one child, E, with a cumulative distance of 10. However, E already exists in our queue with a lesser distance, so we will not add it again.The next minimum distance is that of E, so that is what we will remove|![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/UCS_7.jpg)|
|8|Now, the minimum cost is F, so it will be removed and its child (J) will be added|![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/UCS_8.jpg)|
|9|After this, H has the minimum cost so it will be removed, but it has no children to be added:|![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/UCS_9.jpg)|
|10|Finally, we remove the Dest node, check that it is our target, and stop the algorithm here. The minimum distance between the source and destination nodes (i.e., 8) has been found.|![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/UCS_1.jpg)|


### Performance Measure:

**Completeness**

* Yes, if step cost ≥ ε. Therefore, it will get stuck in an infinite loop if there is a path with an infinite sequence of zero-cost actions.

**Time complexity**

Let C* be cost of the optimal solution, and ε be each step to get closer to the goal node. Then the number of steps is C/ε .
Hence, the worst-case time complexity of Uniform-cost search is O( b^(C*/ε) ).

 
**Space complexity**

The same logic is for space complexity.Number of nodes with f ≤ cost of optimal solution, O(b^⌈C∗/ε⌉)
> O(nd)

**Optimality**

* Uniform-cost search is optimal. This is because, at every step the path with the least cost is chosen, and paths never gets shorter as nodes are added, ensuring that the search expands nodes in the order of their optimal path cost


##  Depth-first search

Depth-first search (**DFS**) is an algorithm for traversing or searching tree or graph data structures. The algorithm starts at the root node (selecting some arbitrary node as the root node in the case of a graph) and explores as far as possible along each branch before backtracking

**Depth First Search Example**

Let's start with a simple example. We use an undirected graph with 5 vertices. (Look at this table below and follow each step)

|Step|Description|Shape|
|----|-----------|-----|
|0|Look at this example and shape.|![](![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/graph-dfs-step-0.webp))|
|1|We start from vertex 0, the DFS algorithm starts by putting it in the Visited list and putting all its adjacent vertices in the stack.( adjacent vertices = {1,2,3} )|![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/graph-dfs-step-1.webp)|
|2|Next, we visit the element at the top of the stack i.e., 1, and go to its adjacent nodes. Since 0 has already been visited, we visit 2 instead.|![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/graph-dfs-step-2.webp)|
|3|Vertex 2 has an unvisited adjacent vertex in 4, so we add that to the top of the stack and visit it. ( we put vertex 4 before vertex 3 in stack)|![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/graph-dfs-step-3.webp)|
|4|Continue Step 3|![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/graph-dfs-step-4.webp)|
|5|After we visit the last element 3, it doesn't have any unvisited adjacent nodes, so we have completed the Depth First Traversal of the graph.|![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/graph-dfs-step-5.webp)|

For more clarification, we give an example like an animation with start to end(look at the gif below).

![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/Depth-First-Search.gif)

### Performance Measure:

**Completeness**

It depends on the search space:
* If the search space is finite, then Depth-First Search is complete as it will expand every node within a limited search tree.
* if there are infinitely many alternatives, it might not find a solution.

**Time complexity**

(Equivalent to the number of nodes traversed in DFS. )
> T(n)= 1 + n +n^2 + n^3 + ... + n^d = O(n^d)

 
**Space complexity**

(Equivalent to how large can the fringe get. )
> O(nd)

**Optimality**

DFS is not optimal, meaning the number of steps in reaching the solution, or the cost spent in reaching it is high. 



## Depth-limited Search

The depth-limited search (DLS) method is almost equal to depth-first search (DFS), but DLS can work on the infinite state space problem because it bounds the depth of the search tree with a predetermined limit L. Nodes at this depth limit are treated as if they had no successors.

Depth-limited search can be terminated with two Conditions of failure:

* Standard failure value: It indicates that problem does not have any solution.
* Cutoff failure value: It defines no solution for the problem within a given depth limit.

Look at the example:


![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/depth-limited-search-algorithm.png)

At first we determine a level until which we are going to expand nodes. For the first step we consider it to be zero. We expand the nodes with DFS startegy and check them if they are goal state or not. If the goal state were not among them, we return to the begininng node and start again but this time we expand the nodes a level deeper which would be level one. We proceed until when we reach out the desired state.

### Performance Measure:

**Completeness**

The limited path introduces another problem which is the case when we choose l < d, in which is our DLS will never reach a goal, in this case we can say that DLS is not complete.

**Time Complexity**

>Time complexity of DLS algorithm is O(b^l)

**Space Complexity**

>Space complexity of DLS algorithm is O(b*l)

**Optimality**

One can view DFS as a special case of the depth DLS, that DFS is DLS with l = infinity.
DLS is not optimal even if l > d.


## Iterative deepening depth-first search

The iterative deepening algorithm is a combination of DFS and BFS algorithms. This search algorithm finds out the best depth limit and does it by gradually increasing the limit until a goal is found.

This algorithm performs depth-first search up to a certain "depth limit", and it keeps increasing the depth limit after each iteration until the goal node is found.

This Search algorithm combines the benefits of Breadth-first search's fast search and depth-first search's memory efficiency.

The iterative search algorithm is useful uninformed search when search space is large, and depth of goal node is unknown.

**Example**

Look at the example below.

![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/IDDFS.jpg)

Suppose that node M is goal. So we start from Limit =0 (node A). We check that A is goal or not, if it is not we add 1 to limit and check the nodes of B and C. So continue until reach the goal.

### Performance Measure:

**Completeness**

This algorithm is complete is if the branching factor is finite.

**Time Complexity**

> T( b ) = (d+1)b^0 + db^1 + (d−1)b^2 + ... + bd = O( b^d )

or more precisely

> O(b^d(1 – 1/b)^-2)

Note: In this algorithm because of the fact that we want to avoid space problems, we won't store any data therefore we may have to repeat some actions but it won't trouble us because time complexity still remains O( b^d ), similar to BFS.


**Space Complexity**

>The space complexity of IDDFS will be O(d), where d is the depth of the goal.

Exactly like DFS, only those nodes will be stored in the stack that represents the branch of the tree which is being expanded. Since the maximum depth of stack is d, the maximum amount of space which is needed would be O(d).

**Optimality**

>IDDFS algorithm is optimal if path cost equals 1.






## Bidirectional search
This algorithm is one of the algorithms that is being used for searching a graph or tree. It is originated from BFS algorithm but with the difference that it starts searching from two nodes and precede to search until those two intersect with each other. This algorithm is so much faster and simpler than the BFS algorithm and using this algorithm is appropriate when all the nodes or at least goal states are defined clearly and the branching factor is exactly the same in both directions. For example, in the chart below. If we start from nodes 0 and 14, these two will reach out to each other on node 7 and then the path will be found.
 
![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/bidir.png)

Now let’s talk about its features and performance measures.

**Completeness**

It is a complete algorithm.

**Optimality**

Again like BFS algorithm, it is not an optimal algorithm in general but if cost of steps be equal to one, this algorithm would be optimal.

**Time and Space Complexity**

It uses memory to save queues and nodes that had been expanded. But the time for reaching an optimal solution is half of the time in BFS algorithm and this is what makes this algorithm so attractive to use.



## Conclusion
The algorithms discussed above are the most used and practical algorithms that are being used in the academic and industry enviornment. These algorithm are being developed and expanded everyday and in a near future we will be using much more faster and efficient algorithms.

![](https://github.com/mohsenosl99/notes/blob/master/notebooks/2_uninformed_search/images/Conclusion.jpg)


The refrences that were used are as follow.

-Artificial Intelligence, A modern approach, Russel & Norvig (Third Edition)

-www.geeksforgeeks.org

-www.javatpoint.com/ai-uninformed-search-algorithms











