# Random Walks - Generating Networks in Networkx



<img src="https://github.com/LawrenceMMStewart/M3C1-Networks-and-Random-Walks/blob/master/Images/plot1.png" width="450">

This project is an attempted solution to the following Imperial College HPC classwork1:
You will now develop code to simulate a new network generation model. First, a seed node is placed at (X,Y)=(0,0). Then, a new node is introduced at (X,Y) = (0,H) where H is a positive input parameter. This new node undergoes a biased 2-D random walk with β=1β=1, α∈{−1,0,1}α∈{−1,0,1} and which terminates when either: 1) the node reaches Y=0 or 2) the node passes within a distance, d∗=1+(1+α)2‾‾‾‾‾‾‾‾‾‾‾‾√d∗=1+(1+α)2 of the first node. In the latter case, node 2 will have linked to node 1. This process is then repeated with new nodes being introduced one at a time at (0,H) and linking when they pass within d≤d∗d≤d∗ of any other nodes in the network (and otherwise stopping at Y=0). Network generation stops when a node is added to the network at height, Y≥HfY≥Hf (with Hf specified as an input parameter). Notes: 1) You may assume that H and Hf are integers with 1<Hf≤H−31<Hf≤H−3; 2) A new node is introduced only when the previous node has completed its random walk and has thus been added to the network.

i) Complete the function rwnet1 and implement this model. The function should return the coordinates of the nodes in the final network in the variables X and Y. When the input variable display is True, a well-made figure should be created which displays the network. Typical parameter values to try are H=500H=500, Hf=300Hf=300 though you may wish to choose smaller heights while developing your code.

ii) Now consider a modified version of this model where vertical ‘walls’ are placed at X=±LX=±L. During a random walk, any step attempting to take a node across a wall should be modified to place the node’s horizontal position at the wall. Add code to rwnet2 so that it implements this modified model, but whose functionality (input, output, figure generation) is otherwise identical to rwnet1.

3. (25 pts) Investigate and analyze the rate at which network height increases during simulations. Fix the heights to be H=200H=200 and Hf=150Hf=150. You should analyze the influence of bias with α=0,1α=0,1 and the influence of walls with L=∞,30,150L=∞,30,150. Add code to the function analyze which generates 1-3 figures which clearly illustrate the most important trends. In the docstring for analyze, clearly and concisely explain: 1) what quantity/quantities are displayed in the figure(s), 2) the trends that you are displaying, and 3) the significance of these trends (i.e. what is causing them and how they effect the networks). Save your figures as .png files with the names hw11.png, hw12.png, hw13.png and submit them with your codes.

4. (25 pts) Now, you will develop a function which converts a set of ‘network coordinates’, X,YX,Y, into a conventional network with numbered nodes and links. Given a set of node coordinates stored in two numpy arrays, XX and YY, a link is placed between any two nodes that are within a distance, d≤d∗d≤d∗, of each other. The function, network, takes XX, YY, and d∗d∗ as input. Nodes that do not link to any other nodes should be discarded. Add code to the function so that it generates the corresponding NetworkX graph and returns it. If the input variable display is true, a figure of the graph should be displayed. If degree is true, the degree distribution for the graph should be computed, displayed, and returned. In the function’s docstring, concisely explain how the graph is created.







# Contents 

* **rw2d(Nt,M,a,b)** - Perform a 2D Random Walk with Left and Right Biases a,b. Nt=Number of steps, M = number of simultations
* **rwnet1(H,Hf,a,display=False)** - Build a network using a random walk of right bias a. H is starting drop height. Hf is termination height.
* **rwnet2(L,H,Hf,a,display=False)** -Build a network using a random walk of right bias a. L is the distance of placed walls from origin. H is starting drop height. Hf is termination height.
* **analyse(H,Hf)** - Produce analysis of rate of increase of varied bias and wall length random walk graphs.
* **network(X,Y,dstar,display=False,degree=False)** - Create a network from co-ordinates X,Y of all nodes that link to another node within distance dstar. Display for a plot of the network. Degree for a degree distribution analysis.





<img src="https://github.com/LawrenceMMStewart/M3C1-Networks-and-Random-Walks/blob/master/Images/plot2.png" width="450">

<img src="https://github.com/LawrenceMMStewart/M3C1-Networks-and-Random-Walks/blob/master/Images/plot3.png" width="450">

<img src="https://github.com/LawrenceMMStewart/M3C1-Networks-and-Random-Walks/blob/master/Images/plot4.png" width="450">

<img src="https://github.com/LawrenceMMStewart/M3C1-Networks-and-Random-Walks/blob/master/Images/plot5.png" width="450">




## License

The code is distributed under a Creative Commons Attribution 4.0 International Public License.



