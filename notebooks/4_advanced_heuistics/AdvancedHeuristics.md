<div style="direction:rtl;line-height:300%;">
	<font face="XB Zar" size=5>
		<div align=center>
			<font face="IranNastaliq" size=30>
				<p></p>
                <b>Advanced Heuristics</b>
<img src="pics/opening.jpg" style="float:center; width:60%"/>
				<p></p>
			</font>
			<font>
By Fatemeh Bahrani, Rosta Roghani, Yasaman Shaykhan
            </font>
			<p></p>
</div>

<p></p>
<br />
<div id="sec_intro" style="line-height:300%;">
	<font face="XB Zar" size=5>
        <font color=#FF7500 size=6>
Heuristic Functions
        </font>
        <br/>
        <hr>
		<font>
            <b>Definition:</b>
        </font>
		<p></p>
This function takes a state of the environment and estimates the shortest path of the node to the target and returns it. In fact, this function is one of the search criteria for selecting estimates of the cost of the route so that it succeeds in reaching the nearest goal. The better h is, the less mistakes we make and the faster we get the answer. The most common way that transfers problem information to a search operation is called a heuristic function, usually denoted by h (n).    
	</font>
</div>

<p></p>
<div id="sec_intro" style="line-height:300%;">
	<font face="XB Zar" size=5>
In the case of heuristic functions, two features are important; Being <b>monotonic</b> and <b>admissible</b>.
        <br/>
&#9679; Being “Admissible” means that the heuristic function underestimates the cost of reaching goal from node.
        <img src="pics/admissable.png" style="float:center; width:60%"/>
	</font>
</div>

<p></p>
<div id="sec_intro" style="line-height:300%;">
	<font face="XB Zar" size=5>
&#9679; Being “Monotonic” means that f values never decrease from node to descendants. 
	</font>
</div>
<br>
<figure class="half" style="display:flex">
    <img src="pics/monotonic1.png" style="width:50%">
    <img src="pics/monotonic2.png" style="width:50%">
</figure>

<p></p>
<br>
<div id="sec_intro" style="line-height:300%;">
	<font face="XB Zar" size=5>
It should be noted that being monotonic is a sufficient condition for being admissible. (Nonetheless, most admissible heuristics are also consistent.) 
        <br>
        <b>Proof:</b>
        <img src="pics/mono-admiss.png" style="float:center; width:90%"/>
	</font>
</div>

<p></p>
 <hr>
<div id="sec_intro" style="line-height:300%;">
	<font face="XB Zar" size=5>
<b>How to make a heuristic function Monotonic (or Consistent):</b>
       <img src="pics/makeconsistant.png" style="float:center; width:90%"/>
    <br>
    In case of Tree Search, only the condition of h being admissible is required.
Two conditions are required for the h function in Graph Search: 1. being Admissible and 2. being Monotonic.
<br><br>
        <font color=red size= 5> <b>&lowast;</b> </font>Finding the optimal h (h&lowast;) is difficult in some cases, and this is an important choice. The closer h is to h&lowast;, the fewer nodes open. If h &le; h*, it will find the optimal answer.If h and h&lowast; are equal, it only opens the path to the optimal answer. But if h &gt; h&lowast; it may not find the optimal answer. 
        <br>
        Counterexample:
         <img src="pics/counter.jpg" style="float:center; width:50%"/>
	</font>
</div>

<br>
<p></p>
<br />
<div id="sec_intro" style="line-height:300%;">
	<font face="XB Zar" size=5>
        <font color=#FF7500 size=6>
Heuristic Dominance
        </font>
        <br/>
        <hr>
		<p></p>
It is used to compare the performance of two heuristic functions.
        <br>
        &#9679; If h<sub>2</sub>  &ge; h<sub>1</sub> for all n (both admissable) then h<sub>2</sub> <font color=red> dominates </font> h<sub>1</sub> <br>
         h<sub>2</sub>is better for search
        <br>
        &#9679; Typical earch costs (average number of nodes axpanded) for 8-puzzle problem
        <br>
        &ensp; &nbsp; &nbsp;
        d = 12 : <br>
        &emsp; &emsp; &emsp; IDS = 3,644,035 nodes <br>
        &emsp; &emsp; &emsp; A&lowast;(h<sub>1</sub>) = 227 nodes <br>
        &emsp; &emsp; &emsp; A&lowast;(h<sub>2</sub>) = 73 nodes <br>
        &ensp; &nbsp; &nbsp;
        d = 24 : <br>
        &emsp; &emsp; &emsp; IDS = too many nodes <br>
        &emsp; &emsp; &emsp; A&lowast;(h<sub>1</sub>) = 39,135 nodes <br>
        &emsp; &emsp; &emsp; A&lowast;(h<sub>2</sub>) = 1,641 nodes
	</font>
</div>

<p></p><hr>
<div id="sec_intro" style="line-height:300%;">
	<font face="XB Zar" size=5>
<b>heuristic function design constraint relaxation:</b>
        <br>
        There are ways to improve the heuristic function. One of these ways is relaxation. In this method, we look for the answer in a space with fewer terms and conditions; and instead of minimizing the cost, we find the lower bound. As a result, it has become easier to solve.
    <img src="pics/constraint.png" style="width:80%">
        In general, admissible heuristic functions represent the cost of exact solutions to simplified or relaxed versions of the original problem (Pearl, 1984). For example, in a sliding-tile puzzle, to move a tile from position x to position y, x and y must be adjacent, and position y must be empty. By ignoring the empty constraint, we get a simplified problem where any tile can move to any adjacent position. We can solve any instance of this new problem optimally by moving each tile along a shortest path to its goal position, counting the number of moves made. The cost of such a solution is exactly the Manhattan distance from the initial state to the goal state. Since we removed a constraint on the moves, any solution to the original problem is also a solution to the simplified problem, and the cost of an optimal solution to the simplified problem is a lower bound on the cost of an optimal solution to the original problem. Thus, any heuristic derived in this way is admissible.
        <br>
        <b>More Relaxed Heuristic Functions</b>:
        <br>
&emsp; &minus; Pattern Database Heuristics <br>
&emsp; &minus; Linear Conflict Heuristics <br>
&emsp; &minus; Gaschnig’s Heuristics
	</font>
</div>
<br>

<br>
<p></p>
<br />
<div id="sec_intro" style="line-height:300%;">
	<font face="XB Zar" size=5>
        <font color=#09C42B size=5.5>
Pattern Database:
        </font>
        <br/>
		<p></p>
We can consider a subset of the search space and consider others as "don't care". In this case, the interaction between the cells inside this subset is considered and the independence is reduced. The function h can then be obtained by combining "h"s from different subsets (search space separation).

Pattern databases are for exploratory estimation of storing state-to-target distances in state space. Their effectiveness depends on the choice of basic patterns. If it is possible to divide the subsets into separate subsets so that each operator only affects the subsets in one subset, then we can have a more acceptable exploration performance. We used this method to improve performance in 15-puzzles with a coefficient of more than 2000 and to find optimal solutions for 50 random samples of 24-puzzles.
        <img src="pics/15puzzle.png" style="float:center; width:80%" />
        
 How do we combine the “h” s of the separated subset of state space?

&emsp; &minus; MAX: Which has diminishing. <br>
&emsp; &minus; ADD: In this case, the limitation is removed and it is admissible. <br>
Using a Pattern Database helps us solve many problems, but it is flawed in very large cases and is not scalable.
        <br><br>
        <font color=#09C42B size=5.5>
            Drawbacks of Pattern DBs:
        </font>
        <br>
            &#9679; Since we can only take max <br>
&emsp; &#9679; Diminishing returns on additional DBs <br>
&emsp; &#9679; Consider bigger problem instances. <br>
&emsp; &emsp; &emsp; &#9679; Subproblems should be small to be scalable. <br>
&emsp; &emsp; &emsp; &#9679; If hi(n) from each database is at most x, then maxi hi would be at most x. <br>
        &#9679; Would like to be able to add values 
	</font>
</div>

<br>
<p></p>
<br />
<div id="sec_intro" style="line-height:300%;">
	<font face="XB Zar" size=5>
        <font color=#09C42B size=5.5>
Disjoint of Pattern DBs:
        </font>
        <br/>
		<p></p>
What if we make patterns be disjoint sets? Can we take summation of heuristics by pattern DBs as an admissible heuristic? In order to fix this, take number of moves made to the specified tiles as hi instead.
        <img src="pics/disjoint.png" style="float:center; width:80%" />
        
Why does summation result in an admissible heuristic in that case?
        <br>
        <b> Proof: </b>
         <img src="pics/proof.png" style="float:center; width:50%" />

Manhattan dist. is a trivial example of a disjoint DBs, where each group contains only a single tile. 
As a general rule, when partitioning the tiles, we want to group together tiles that are near each other in the goal state, since these tiles will interact the most with one another. 

Using this method, the 15-puzzle problem is solved 2000 times and the 24-puzzle problem is 12 million times faster
	</font>
</div>

<p></p>
<br/>
<div id="sec_refs" style="line-height:300%;">
	<font face="XB Zar" size=5>
		<font color=#FF7500 size=6>
Resources
        </font>
		<hr>       
        <ul>
            <li>
           www.cs.stackexchange.com/questions/63481/how-does-consistency-imply-that-a-heuristic-is-also-admissible
            </li>
            <li>
            www.courses.cs.washington.edu/courses/cse473/12sp/slides/04-heuristics.pdf           
            </li>
            <li>
            www.sciencedirect.com/science/article/pii/S0004370201000923
            </li>
            <li>
            www.researchgate.net/publication/222830183_Disjoint_pattern_database_heuristics
            </li>
              <li>
            www.aaai.org/Papers/JAIR/Vol22/JAIR-2209.pdf
            </li>
             <li>
            www.link.springer.com/chapter/10.1007/978-3-540-74128-2_3
            </li>
                        <li>
               www.stackoverflow.com/questions/46554459/intuitively-understanding-why-consistency-is-required-for-optimality-in-a-searc
            </li>
        </ul>
	</font>
</div>
