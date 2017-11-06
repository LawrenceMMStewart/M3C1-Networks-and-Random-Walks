"""M3C 2017 Homework 1 - Lawrence Stewart CID 00948972 -email ls3914@ic.ac.uk 
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def rw2d(Nt,M,a=0,b=0):
    """Input variables
    Nt: number of time steps
    M: Number of simulations
    a,b: bias parameters
    """

    #Ensure the constraints on a,b given in the question
    assert a>=-1, "a must be greater than or equal to -1"
    assert 1>=b, "b must be less than or equal to 1"

    # Generate Nt randomly selected values for both the X and Y co-ordinate with x in {-1,1+a} and y in {-1,1-b}
    Xmat=np.random.choice(np.array([-1,1+a]),[Nt+1,M])
    Ymat=np.random.choice(np.array([-1,1-b]),[Nt+1,M])

    #Set initial conditions for Nt=0:
    Xmat[0,:]=0.
    Ymat[0,:]=0.

    # If errors occur change back to this:
    # Xmat2=np.cumsum(Xmat,axis=0)
    # Ymat2=np.cumsum(Ymat,axis=0) 

    Xmat=np.cumsum(Xmat,axis=0)
    Ymat=np.cumsum(Ymat,axis=0)

    #Generate <X>,<Y>:
    X_=np.mean(Xmat,axis=1)
    Y_=np.mean(Ymat,axis=1)

    #Generate <X^2>, <Y^2>
    X2_=np.mean(np.multiply(Xmat,Xmat),axis=1)
    Y2_=np.mean(np.multiply(Ymat,Ymat),axis=1)

    #Generate <XY>
    XY_=np.mean(np.multiply(Xmat,Ymat),axis=1)
    
    return X_,Y_, X2_, Y2_, XY_


def rwnet1(H,Hf,a=0,display=False):
    """Input variables
    H: Height at which new nodes are initially introduced
    Hf: Final network height
    a: horizontal bias parameter, should be -1, 0, or 1
    display: figure displaying the network is created when true
    Output variables
    X,Y: Final node coordinates
    output: a tuple containing any other information you would
    like the function to return, may be left empty
    """

    #Principlal idea behind method: At each random walk, calculate the number of steps in any direction that can be done at once
    # without violating any of the conditions. Vectorise the process for all of these steps, then re-evaluate.

    #Let output be the y co-ordinate if the highest node in the network
    output=np.array([0])

    #Ensure the constraints on a, given in the question
    assert a in [-1,0,1], "Input error - a must be either 1, 0, or -1" 

    #Ensure constraints placed on H, given in the question
    assert H>=0, "Input error - H must be positive"

    #Ensure the constraints placed on Hf
    assert Hf>1, "Hf must be an integer greater than 1"
    assert H+3>=Hf, "Hf must be less than or equal to H-3"


    #Variable to check that no value has been placed above Hf
    place_check=False

    #Variable to check that node has not violated either of the conditions for continued random walk
    walk_check=False

    #List of the node co-ordinates (x,y), where we start with the origin node (0,0)
    X,Y=np.array([0]),np.array([0])



    #Calculate the sqrt(1^2+(1+alpha)^2) which will the upper bound for a single movement step in optimal direction during a walk
    maxstep=np.sqrt(1+(1+a)**2)


    while place_check==False:

        #Initiate new node placement, x y are the current co-ordinates of the random walk:
        x_cur,y_cur=0,H

        #Reset Walk_check to False, for we are starting a new walk:
        walk_check=False
      
        #If walk_check remains false begin a new random walk:
        while walk_check==False:
          
            #Calculate the distance to the closest neighbour and to the y axis
            xdist,ydist=X-x_cur,Y-y_cur
            node_dist=np.sqrt(np.multiply(xdist,xdist)+np.multiply(ydist,ydist)) #This generates the distance of current point from all nodes
            min_ind=np.argmin(np.array([min(node_dist),y_cur])) 

            #If min_ind=0 then the closest thing is a node and if min_ind=1 then the closest thing is the y axis

            #set the distance (if a node is the closest)
            if min_ind==0:
                d=min(node_dist)

            #set distance (if the y=0 axis is the closest)
            else:
                d=y_cur

            #Calculate the maximum steps we can do as one process safely i.e floor( d/ maxstep) [note int() rounds down]
            nskip=int(d/maxstep)
            

            #Check if nskip=0 (in which case we are have finished the walk)
            if nskip==0:
                
                #Add node co-ordinates
                X=np.append(X,x_cur)
                Y=np.append(Y,y_cur)

                #complete the walk:
                walk_check=True

                #Update the height of the network:
                output=np.append(output,max(Y))

                #If termination condition has been reached - end simulation
                if y_cur>=Hf:
                    place_check=True

            #Perform random walk for nskip iterations:

            #1) As we are doing nskip steps in one we can sample from binomial distribution for both X and Y and 
            #sum the total successes, where p success =1/2  (a success will be the # of -1's while a fail will be the number of 1+a or 1-b's)

            x_rand=sum(np.random.binomial(1,0.5,nskip))
            y_rand=sum(np.random.binomial(1,0.5,nskip))

            #we have x_rand numbers of -1 steps and (nskip-xrand) number of (1+a) steps
            x_cur=x_cur+((1+a)*(nskip-x_rand)-x_rand)

            #we have y_rand number of -1 steps and (n_skip-y_rand) number of (1-b=0) steps
            y_cur=y_cur-y_rand


    #Create Plots if User Desires
    if display==True:
        plt.figure()
        plt.suptitle('Lawrence Stewart - Created Using rwnet1().')
        plt.scatter(X,Y,c=X,cmap='viridis_r')
        plt.xlabel("x co-ordinates")
        plt.ylabel("y co-ordinates")
        plt.title("Plot of Generated Network with a=%i"%a )
        plt.grid()
        plt.show()

    return X,Y,output

def rwnet2(L,H,Hf,a=0,display=False):
    """Input variables
    L: Walls are placed at X = +/- L
    H: Height at which new nodes are initially introduced
    Hf: Final network height
    a: horizontal bias parameter, should be -1, 0, or 1
    display: figure displaying the network is created when true
    Output variables
    X,Y: Final node coordinates
    output: a tuple containing any other information you would
    like the function to return, may be left empty
    """
    #Principlal idea behind method: At each random walk, calculate the number of steps in any direction that can be done at once
    # without violating any of the conditions. Vectorise the process for all of these steps, then re-evaluate.

    #Ensure the constraints on a, given in the question
    assert a in [-1,0,1], "Input error - a must be either 1, 0, or -1" 

    #Ensure constraints placed on H, given in the question
    assert H>=0, "Input error - H must be positive"

    #Ensure the constraints placed on Hf
    assert Hf>1, "Hf must be an integer greater than 1"
    assert H+3>=Hf, "Hf must be less than or equal to H-3"

    #Let output be the y co-ordinate if the highest node in the network
    output=np.array([0])

    #Variable to check that no value has been placed above Hf
    place_check=False

    #Variable to check that node has not violated either of the conditions for continued random walk
    walk_check=False

    #List of the node co-ordinates (x,y), where we start with the origin node (0,0)
    X,Y=np.array([0]),np.array([0])

    #Calculate the sqrt(1^2+(1+alpha)^2) which will the upper bound for a single movement step in optimal direction during a walk
    maxstep=np.sqrt(1+(1+a)**2)


    while place_check==False:
        
        
        #Initiate new node placement, x y are the current co-ordinates of the random walk:
        x_cur,y_cur=0,H

        #Reset Walk_check to False, for we are starting a new walk:
        walk_check=False
      
        #If walk_check remains false begin a new random walk:
        while walk_check==False:
          
            #Calculate the distance to the closest neighbour and to the y axis
            xdist,ydist=X-x_cur,Y-y_cur
            node_dist=np.sqrt(np.multiply(xdist,xdist)+np.multiply(ydist,ydist)) #This generates the distance of current point from all nodes
            min_ind=np.argmin(np.array([min(node_dist),y_cur])) 

            #If min_ind=0 then the closest thing is a node and if min_ind=1 then the closest thing is the y axis

            #set the distance (if a node is the closest)
            if min_ind==0:
                d=min(node_dist)

            #set distance (if the y=0 axis is the closest)
            else:
                d=y_cur

            #calculate the distance to the walls
            walld=min(abs(L-x_cur),abs(-L-x_cur))

            #Calculate the minimum number of steps that we have until hitting a wall
            wallskip=int(walld/max( 1,abs(1+a) ) )

            #take into account walls:
            
            #Calculate the maximum steps we can do without hitting Y=0 or another node
            nskip=int(d/maxstep)

            #Check if nskip=0 (in which case we are have finished the walk)
            if nskip==0:
                
                #Add node co-ordinates
                X=np.append(X,x_cur)
                Y=np.append(Y,y_cur)

                #complete the walk:
                walk_check=True

                #Update the height of the network:
                output=np.append(output,max(Y))

                #If full termination condition has been reached - end simulation
                if y_cur>=Hf:
                    place_check=True
                break

            #skip is the number of steps we can do at once
            skip=max(1,min(nskip,wallskip))

            #Perform random walk for skip iterations:

            #1) As we are doing skip steps in one we can sample from binomial distribution for both X and Y and 
            #sum the total successes, where p success =1/2  (a success will be the # of -1's while a fail will be the number of 1+a or 1-b's)

            x_rand=sum(np.random.binomial(1,0.5,skip))
            y_rand=sum(np.random.binomial(1,0.5,skip))

            #we have x_rand numbers of -1 steps and (skip-xrand) number of (1+a) steps
            x_cur=x_cur+((1+a)*(skip-x_rand)-x_rand)

            #we have y_rand number of -1 steps and (skip-y_rand) number of (1-b=0) steps
            y_cur=y_cur-y_rand

            #check to see if node has gone past the walls:
            if x_cur<-L:
                x_cur=-L
               
            if x_cur>L:
                x_cur=L


    #Create Plots if User Desires
    if display==True:
        plt.figure()
        plt.suptitle('Lawrence Stewart - Created Using rwnet2().')
        plt.scatter(X,Y,c=X,cmap='viridis_r')
        plt.xlabel("x co-ordinates")
        plt.ylabel("y co-ordinates")
        plt.title("Plot of Generated Network with a=%i with walls at L=%i"%(a,L) ) #This line could need work
        plt.grid()
        plt.show()

    return X,Y,output
    
def analyze(H,Hf):
    """ Analysis: 
 
    hw11.png, (which should contain a subplot of two graphs) shows the rate at which the Network Height Increases, with one plot for 
    the unbiased scenario a=0, and the other plot for the biased scenario a=1. The independant variable is the current height of the network/number of
    nodes placed. Both plots show this for walls setup at L=30, 150 and infinity. Examining the first graph (a=0 unbiased), it can be seen that location of the walls
    has little influence on how the rate of increase of height of the network. It is also worth noting that for each of the 3 values of L, we have that the network 
    stops being built after approximately 2500 iterations (where an iteration is the placement of a node.) 

    However, when we examine the biased a=1 example, we see very different behaviour for different values of L. For L=30 we see that the network reaches height Hf 
    within a very fast time -approximately 150 iterations. For L=150 and L=inf, we similary see the network is reaches termination height much faster, after 
    approximately 900 and 1500 iterations, respectively (compare these three values to the 2500 iterations required to reach Hf in the unbiased case). 

    Apart from an initial spike (in which the first few nodes are placed), the rate of growth for the unbiased test (a=0) is approx 0.07 for each value of L. 
    For the biased case (a=1) we see that for L=30, the rate is approximately 1.3 on average,  implying that the network height growing very fast. For L=150, the 
    rate is slightly lower at roughly 1.1, but then steadily decays with iterations. For L=infinity, we do not see the same high initial rate as with L=30,150,
    and the rate remains fairly constant (however it still takes far less iterations to reach Hf).

    This should be expected, for the expected position for a unbiased 1D random walk is the starting position. In our system, the starting point is (0,H). 
    Considering only the movement in the X direction, with a=0 we should expect the majority of nodes to be near X=0. This means it is far more unlikely that nodes
    will reach X=L. Hence, the different distances for wall placements do not have much effect on the rate of increase of height of the network. This leaves the 
    simulations with L=30,150,inf to have similar results for the unbiased walk, as they are undergoing a similar stochiastic process.

    With the bias added (a=1), it is expected for the majority of nodes to be shifted to the right. With L=30, it is very likely that nodes will reach the wall.
    Similary (but less likely), for L=150. The clumping of nodes close to the wall results in a higher chance of a node finishing the random walk earlier, which
    results in a higher chance of the node increasing the height of the network (as nodes can either move down or stay at the same height). With L=inf we do not
    see the same explosive growth of the network as no walls are present. However, the spread of nodes will be biased to one direction, hence the networks height will
    still grow faster than in the unbiased case. Furthermore, the distance that is required for a node to link and finish the random walk is larger with a=1 dstar=sqrt(5)
    compared to a=0 (dstar= sqrt(2)).
    
    hw12.png shows the height of the network vs the number of nodes placed for the 6 different combinations of a and L. As expected, for the reasons discussed above,
    we see that L=30 a=1 has the largest rate of increase of height, followed by L=150 a=1. The figure helps highlight the difference between the L=inf a=1 case vs the
    unbiased simulations. We see that despite the lack of walls, the biased network still grows at a rate substantially larger than the unbiased cases (for reasons 
    discussed above.)

    """ 


    #Create plots for heights for all combinations for h 

    print("------Generating Graphs -----",end='\r')
    _,_,Ha0Linf=rwnet1(H,Hf,0)
    print("-----Built Graph 1/6-----",end='\r')

    _,_,Ha1Linf=rwnet1(H,Hf,1)
    print("-----Built Graph 2/6-----",end='\r')

    _,_,Ha0L30=rwnet2(30,H,Hf,0)
    print("-----Built Graph 3/6-----",end='\r')

    _,_,Ha1L30=rwnet2(30,H,Hf,1)
    print("-----Built Graph 4/6-----",end='\r')

    _,_,Ha0L150=rwnet2(150,H,Hf,0)
    print("-----Built Graph 5/6-----",end='\r')

    _,_,Ha1L150=rwnet2(150,H,Hf,1)
    print("-----Built Graph 6/6-----",end='\r')

    print("-----Process Complete-----",end='\r')
    
    #Generate the number of nodes for each of the simulations
    nodes_Ha0Linf=np.linspace(1,np.size(Ha0Linf),np.size(Ha0Linf))
    nodes_Ha1Linf=np.linspace(1,np.size(Ha1Linf),np.size(Ha1Linf))
    nodes_Ha0L30=np.linspace(1,np.size(Ha0L30),np.size(Ha0L30))
    nodes_Ha1L30=np.linspace(1,np.size(Ha1L30),np.size(Ha1L30))
    nodes_Ha0L150=np.linspace(1,np.size(Ha0L150),np.size(Ha0L150))
    nodes_Ha1L150=np.linspace(1,np.size(Ha1L150),np.size(Ha1L150))


    #Plotting
    plt.figure(figsize=(14, 6))  
    plt.suptitle('Lawrence Stewart - Created Using analyze().')

    #Plot a=0 graphs for H/#nodes i.e the rate with #of nodes
    plt.subplot(121)
    plt.plot(nodes_Ha0L30,Ha0L30/nodes_Ha0L30, 'g-', label="L = 30",alpha=0.7)
    plt.plot(nodes_Ha0L150,Ha0L150/nodes_Ha0L150, 'r-', label="L = 150",alpha=0.7)
    plt.plot(nodes_Ha0Linf,Ha0Linf/nodes_Ha0Linf, 'c-', label="L = Inf",alpha=0.7)
    plt.xlabel("Number of Nodes Placed")
    plt.ylabel("Height of Network /Number of Nodes")
    plt.title("Rate at which Network Height Increases for varied L, a=0")
    ax = plt.gca()
    ax.set_facecolor('#D9E6E8')
    plt.grid('on')
    plt.legend()


    # #plot a=1 graphs for H/#nodes i.e the rate with #of nodes
    plt.subplot(122)
    plt.plot(nodes_Ha1Linf,Ha1Linf/nodes_Ha1Linf, 'c-', label="L = Inf")
    plt.plot(nodes_Ha1L150,Ha1L150/nodes_Ha1L150, 'r-', label="L = 150")
    plt.plot(nodes_Ha1L30,Ha1L30/nodes_Ha1L30, 'g-', label="L = 30")
    plt.xlabel("Number of Nodes Placed")
    plt.ylabel("Height of Network /Number of Nodes")
    plt.title("Rate at which Network Height Increases for varied L, a=1")
    plt.grid('on')
    ax = plt.gca()
    ax.set_facecolor('#D9E6E8')
    plt.legend()
    plt.show()

    
    #Plot the heights vs the number of nodes for all combinations:
    plt.figure(figsize=(9, 7)) 
    plt.suptitle('Lawrence Stewart - Created Using analyze().')
    plt.plot(nodes_Ha0L30,Ha0L30, label="L = 30, a=0",alpha=0.7)
    plt.plot(nodes_Ha0L150,Ha0L150, label="L = 150, a=0",alpha=0.7)
    plt.plot(nodes_Ha0Linf,Ha0Linf, label="L = Inf, a=0",alpha=0.7) #need to edit the colors
    plt.plot(nodes_Ha1Linf,Ha1Linf, label="L = Inf, a=1",alpha=0.7)
    plt.plot(nodes_Ha1L150,Ha1L150, label="L = 150, a=1",alpha=0.7)
    plt.plot(nodes_Ha1L30,Ha1L30, label="L = 30, a=1",alpha=0.7)
    plt.xlabel("Number of Nodes Placed")
    plt.ylabel("Height of Network")
    plt.title("Development of Network Height with Node Placement")
    ax = plt.gca()
    ax.set_facecolor('#D9E6E8')
    plt.grid('on')
    plt.legend()
    plt.plot()
    plt.show()
    return None


def network(X,Y,dstar,display=False,degree=False):
    """ Input variables
    X,Y: Numpy arrays containing network node coordinates
    dstar2: Links are placed between nodes within a distance, d<=dstar of each other
        and dstar2 = dstar*dstar
    display: Draw graph when true
    degree: Compute, display and return degree distribution for graph when true
    Output variables:
    G: NetworkX graph corresponding to X,Y,dstar
    D: degree distribution, only returned when degree is true
    """

    """
    The function network() takes in co-ordinates (x,y) in the form of two numpy arrays X,Y. The function then loops through the arrays and compares X[i] to
    all X[j] where j>i (and does the same for Y) checking for the condition that d((x_i,y_i),(x_j,y_j))=<dstar. If this condition is met, the nodes will be linked and
     a tuple denoting the link will be added to a list that contains all the edges to be formed. From here the function uses NetworkX's inbuilt function add_edges_from() 
     to build the graph. Finally plots are returned for the graph and degree distribution if the user requires. 


    Analysis: It is evident that the network's maximum degree is constrained to a small value. This is because as soon as a node becomes within a proximity
    of dstar of another node, the random walk is terminated and the a connection is formed between the two nodes. Suppose for example some nodes have formed a
    "cluster" of some form. The probability that a new node penetrates into this cluster is far smaller than the probability that the node will link to 
    the outside of the cluster. Inductively we can see that it is very unlikely that we will achieve high degrees in the network. 

    A simple solution would be to slightly modify the function network(). Once a node is placed and a link is formed with another node
    we could then form links with all nodes in a ball of radius R (where R would be an additional input of network()). A large R value would vastly increase the maximum
    degree reached.

    A more technical change would be to alter the functionality of rwnet1 and rwnet2, so that both functions take in an additional parameter Nlink, where Nlink
    is the number of nodes that a node taking a random walk must come with a distance of dstar of to terminate the walk, i.e placing the node in the network and 
    forming links. This would increase the probability of nodes penetrating deeper into clusters, and reduce the probability of the nodes attatching to the outside 
    of these clusters. A problem with this method would be the placements of the first nodes, which would all end up on the Y=0 axis. A possible solution to this would
    be to introduce a dynamic value of Nlink. This could be either an array or possibly a function that maps the current iteration number to a value for Nlink. This would
    allow the simulation to initially run exactly the same is in our current code, i.e. Nlink has an initial value of 1 for the first K iterations however, after more nodes
    have been placed Nlink increases to allow nodes to form connections of higher degrees. 


    """
    #Create the graph
    G=nx.Graph()


    #preallocate the edges list (and remove possible duplicates in input)
    edges=[]
    X,Y=(zip(*list(set(list(zip(X,Y) ) ))))
    X,Y=np.array(X),np.array(Y)

    for i in range(len(X)-1):

        #Calculate the distance
    
        tempX=X[i+1:]-X[i]
        tempY=Y[i+1:]-Y[i]
        tempd=np.sqrt(np.multiply(tempX,tempX)+np.multiply(tempY,tempY))

        #find all nodes within d*
        ind=np.where(tempd<=dstar)[0]+i+1   

        #add the edges to the list
        edges_to_add=[(i,j) for j in ind] # old one list(zip(i*np.ones(np.size(ind)), ind))
        edges=edges+edges_to_add

    #Flatten list
    checker=list(sum(edges, ()))

    #Take all the nodes that have edges
    nodes_to_add=np.unique(checker) 

    #add them with positions
    for i in range(np.size(X)):
        if i in checker:
            G.add_node(i, pos = (X[i], Y[i]))

    G.add_edges_from(edges)


    #Plot graph if required
    if display==True:
        plt.figure(figsize=(9, 7)) 
        plt.suptitle('Lawrence Stewart - Created using network().')
        # pos=nx.get_node_attributes(G,'pos')
        nx.draw_networkx(G, pos = nx.get_node_attributes(G, 'pos'), with_labels = False, node_size =20, width = 2., font_size = 6,node_color='b',prog='dot')
        # nx.draw(G,pos,node_shape='.',node_size=100, 
        # prog='dot') #add this in when done  cmap=plt.cm.Blues,node_color=range(len(G)),prog='dot'
        node_color=range(len(G)),
        ax = plt.gca()
        ax.set_facecolor('#D9E6E8')
        plt.title("Graph of Network")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid('on')
        plt.show()

    #Plot degree distribution if required
    if degree==True:
        plt.figure(figsize=(9, 7))         
        plt.suptitle('Lawrence Stewart - Created using network().')
        degreeDistribution = nx.degree_histogram(G)
        print(degreeDistribution)
        plt.bar(np.arange(1,len(degreeDistribution)+1,1),[degreeDistribution[i]/len(G.nodes()) for i in range(len(degreeDistribution))],color='b')
        ax = plt.gca()
        ax.set_facecolor('#D9E6E8')
        plt.xticks(np.arange(1,len(degreeDistribution)+1,1))
        plt.title("Degree Distribution for Network Generated")
        plt.grid("on")
        plt.xlabel("Degree")
        plt.ylabel("Node Fraction")
        plt.show()

    return None


if __name__ == '__main__':
    #The code here should call analyze and generate the
    #figures that you are submitting with your code
    analyze(200,150)

 
    



