---
layout: post
title: A short introduction to Artificial Intelligence and Its Branches
author: Hussain A.
categories: [tutorial]
mathjax: true
summary: Branches of Artificial Intelligence
---



## Branches of Artificial Intelligence

Hi there! Artificial Intelligence can be described as the study of intelligent agents. The field of AI research was started from 1956 at Dartmouth college \cite{AIstart}. AI has been widely applied by modern machines for applications such as competing at the highest level in strategic games (e.g. Go, Chess), understanding human speech, autonomous vehicles, and intelligent network routing in content delivery networks etc. Figure below gives an overview of the terms and techniques within the AI research area. Some of the most widely used AI techniques are: Heuristic Techniques, Robotics, Swarm Intelligence, Expert Systems, Turning Test, Logical AI, Planning, Schedule and Optimization, Natural Language Processing, Game Theoretic Learning, Evolutionary Algorithms, Inference, Fuzzy Logic, and Machine Learning etc. We will briefly discuss these techniques in the next subsections.



![]({{ "assets/img/posts/AI.png" | absolute_url }}). 


# Swarm intelligence
Swarm intelligence can be described as a collective behavior  of self-organized and decentralized systems. The term Swarm Intelligence (SI) was first introduced by G. Beni and J. Wang in 1989 for the cellular robotic systems. In the context of Vehicle-to-Everything (V2X) paradigm, the swarm intelligence is shown by a population of vehicular agents that interact locally with one another and their environment. The vehicles follow simple rules without any centralized control system. The behavior of ant colonies, flocks of birds, schools of fish, animal herding, microbial intelligence, and bacterial growth all are based on swarm intelligence. In an experiment performed by Deneubourg in 1990, a group of ants was given two paths (short/long) that connect their nest to the food location. It was discovered from their results that ant colonies had a high probability to collectively select the shortest path. More detail on Swarm Intelligence for wireless communication. The most widely accepted use cases of Swarm Intelligence are 1) Particle Swarm Optimization (PSO), 2) Ant Colony Optimization (ACO) and 3) Swarmcasting.

# Particle Swarm Optimization
PSO is regarded as a global optimization algorithm. It can be used to solve a problem whose solution can be described as a point or a surface in an n-dimensional space. Seeded with an initial velocity, various potential solutions are plotted in this solution space. The particles move around this space with certain fitness criteria, and with the passage of time, particles accelerate towards those locations that have better fitness values. 

# Ant Colony Optimization
ACO was first proposed by M. Dorigo. ACO finds near optimal solutions to different problems which can be described as graph optimization problems. Just as stated earlier, the ants in ACO try to find the shortest path. A famous application in wireless communication routing is known as AntNet. In this routing, near optimal routes are selected without global information.



# Swarmcasting
It exploits the concept of distributed content downloading to provide high resolution video, audio and Peer-to-Peer (P2P) data streams, which contributes in reducing the required bandwidth. It applies the Swarm Intelligence to break down large data into small parts, so that the system can download these parts from different machines simultaneously, which enables a user to start watching the video before downloading is complete. 

## Machine Learning
It covers a big part of AI. ML techniques can be described into three types: Unsupervised learning, Supervised learning, and Reinforcement learning. There are some other kinds of ML schemes such as Transfer learning and Online learning, which can be subcategorized in the form of these basic three ML schemes. ML basically consists of two important stages: training and testing. Based on the realistic data, a model is trained in the training phase. Then in the testing phase predictions are made based on the trained model.

# Unsupervised ML
In the unsupervised learning, training is based on unlabeled data. This scheme tries to find an efficient representation of the unlabeled data. For example, the features of a data can be captured by some hidden variables, that can be represented by the Bayesian learning techniques. Clustering is a form of unsupervised learning that groups samples with similar features. The input features of each data point can be its absolute description or a relative similarity level with other data points. In the wireless networks paradigm, the cluster formation for the hierarchical protocols is of great importance in terms of energy-management, where each member just needs to communicate with the cluster head before communicating with the members of other clusters. Some traditional clustering algorithms are k-means, spectrum clustering, and hierarchical clustering. Dimension reduction is another subclass of unsupervised ML scheme. The main idea behind dimension reduction is to down-sample the data from a higher dimension to a lower dimension without any significant data loss. Applying machine learning for most applications require dimension reduction due to a number of reasons.

Curse of data dimensionality is the first reason. In clustering, classification and optimization, the overall model complexity increases dramatically with the increase in feature dimensions. The second reason is the hurdle in the learning process. In most of the cases the features of the data samples are correlated in some aspects, but if the feature value is affected by noise or interference then the respective outcome of the correlation will be corrupted and the learning process will be affected. Such kind of dimension reduction in the vehicular social networks is the formation which leads to a vehicular cluster. The cluster head collects and transmits the information to the eNodeB to reduce the communication cost. The curse of dimensionality can be reduced by the dimension reduction methods. Dimension reduction methods are grouped in two categories: 1) linear projection methods such as: Principal Component Analysis (PCA) and Singular Value Decomposition (SVD),  and 2) nonlinear projection methods such as: manifold learning, local linear embedding (LLE) and isometering mapping.

# Supervised learning
The supervised learning learns from a set of labeled data. Supervised learning can be divided into two categories: 1) Regression, and 2) Classification. If the training data only includes discrete values then it is a classification problem and the output of the trained model is a classification which is also discrete. On the other hand if the training data contain continuous values, then it is the regression problem and the output of the trained model will be a prediction. Two widely used examples of supervised ML are Decision Trees and Random Forest.
	
The output of regression algorithms is a continuous value that may represent prediction of the house price, stock exchange, banking customer transactions, State Of Charge (SOC) of an electric vehicle battery, level of traffic congestion at various intersections, and jamming prediction. In vehicular social networks, regression can be used to predict parameters such as network throughput. Two classic regression algorithms are: 1) Gaussian Process for Regression (GPR) and 2) Support Vector Regression (SVR). In vehicular networks the classification algorithms can be used for intrusion or malfunction detection. Moreover, classification algorithms are also beneficial in the traffic safety applications such as: Augmented Reality Head Up Display (AR-HUD), active driver information systems, obstacle detection, and predicting complex traffic types.


# Reinforcement Learning 
It actively learns from the actions of the learning agent from the corresponding reward. It means in order to maximize the reward, inexplicit mapping the situations according to the actions by trial and error. The Markov Decision Process (MDP) is an example of reinforcement learning. Q function model-free learning process is a classic example to solve MDP optimization problem that does not require information about the learning environment. Actions and their rewards generate policies of the choice of a proper action. In a given state the Q function estimates the mean of the sum reward. The best Q function is the maximum expected sum reward that can be achieved by following any of the policies. Reinforcement leaning is a perfect candidate for addressing various research challenges in vehicular networks. For example, cooperative optimization of fuel consumption for a given geographical region, handling the spatial and temporal variations of the V2V communications, optimum path prediction of electric vehicles, and reduction in traffic congestions.

# Deep learning 
It is closely related to the above three categories of ML. It is a deeper network of neurons in multiple layers. It aims to extract knowledge from the data representations, that can be generated from the previously discussed three categories of ML. The network consists of input layer, hidden layers and an output layer. Each neuron has a non linear transformation function, such as ReLU, tanh, sigmoid, and leaky-ReLU. The scaling of input data is very crucial as this can severely affect the prediction or classification of a network. As the number of hidden layers increases, the ability of the network to learn also increases. However, after a certain point, any increase in the hidden layers gives no improvement in the performance. The training of a deeper network is also challenging because it requires extensive computational resources, and the gradients of the networks may explode or vanish. The deployment of these resource hungry deeper networks has raised the importance of edge computing technology. Vehicles on the move can get benefit from mobile edge computing servers.

# Expert systems: 
They Emulate the human ability to make decisions. The Expert systems solve complex problems by reasoning which is extracted from human knowledge. This reasoning is represented by IF-THEN rules, instead of procedural coding. An Expert system is divided into two parts: Knowledge base and Inference engine. 1) Knowledge base: The knowledge base is composed of rules extracted from human knowledge. Inference engine: 2) The inference engine applies the extracted rules from knowledge base to known facts to deduce new facts. They can also include explanation and debugging abilities. There are further two modes of an inference engine such as  forward chaining and backward chaining.

# Planning, Scheduling, and optimization: 
This branch of AI deals with the realization of strategies or action sequences, for the execution by intelligent agents. The planning is also related to decision theory, and unlike traditional control and classification problems, the complex solutions must be discovered in an optimized manner from the n-dimensional space. Planning can be performed offline in a known environment with available models, and the solutions can be evaluated prior to the execution. In a highly dynamic and partially known environment of V2X paradigm, the strategy needs to be revised online. The models and related policies must be adapted accordingly. The languages used to describe the scheduling are known as action languages. There are different algorithms of planning, such as Classical planning, Temporal planning, Probabilistic planning, Preference-based planning, and Conditional planning.
