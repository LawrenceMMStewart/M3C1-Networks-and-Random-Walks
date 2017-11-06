# Random Walks - Generating Networks in Networkx



<img src="https://github.com/LawrenceMMStewart/M3C1-Networks-and-Random-Walks/blob/master/Images/plot1.png" width="450"> <img src="https://github.com/LawrenceMMStewart/M3C1-Networks-and-Random-Walks/blob/master/Images/plot2.png" width="450">

This project is an attempted solution to the following Imperial College HPC classwork1: https://imperialhpsc.bitbucket.io/hw/hw1.html#hw1 .

The simulation is based upon a unbiased and biased random walk. We select a height H and a termination height Hf. A node is placed at (0,H) on the axis. A random walk occurs until either a node comes within distance of dstar of another node, where dstar=sqrt(1+bias^2). If this occurs then the two nodes are connected and the node is placed down. If the node reaches the Y=0 axis then the node is placed. Furthermore, walls at a distance +L and -L from the origin can be placed to restrict x movement of the particles. A graph is built when a node is placed above or at the termination height Hf. The generated networks depend on both L and the bias a.

<img src="https://github.com/LawrenceMMStewart/M3C1-Networks-and-Random-Walks/blob/master/Images/plot3.png" width="450">

<img src="https://github.com/LawrenceMMStewart/M3C1-Networks-and-Random-Walks/blob/master/Images/plot4.png" width="450">






# Contents 

* **rw2d(Nt,M,a,b)** - Perform a 2D Random Walk with Left and Right Biases a,b. Nt=Number of steps, M = number of simultations
* **rwnet1(H,Hf,a,display=False)** - Build a network using a random walk of right bias a. H is starting drop height. Hf is termination height.
* **rwnet2(L,H,Hf,a,display=False)** -Build a network using a random walk of right bias a. L is the distance of placed walls from origin. H is starting drop height. Hf is termination height.
* **analyse(H,Hf)** - Produce analysis of rate of increase of varied bias and wall length random walk graphs.
* **network(X,Y,dstar,display=False,degree=False)** - Create a network from co-ordinates X,Y of all nodes that link to another node within distance dstar. Display for a plot of the network. Degree for a degree distribution analysis.


##Affects of L and A on rate of network growth


<img src="https://github.com/LawrenceMMStewart/M3C1-Networks-and-Random-Walks/blob/master/Images/plot5.png" width="500">




## License

The code is distributed under a Creative Commons Attribution 4.0 International Public License.



