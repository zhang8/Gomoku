# Homework 2

Your task in this homework is to implement the Monte Carlo Tree Search used in AlphaZero for the board game Gomoku (five in a line). For simplicity, we consider only the Gomoku game with a 11x11 board. You should implement a class named "MCTS" which must be a subclass of the class MCTSBase. The MCTSBase class already implemented the overall search process. In your MCTS class, you need to fill in the missing components by implementing (override) the abstract methods. Your code also needs to implement another class (use any class name you want) which must be a subclass of the TreeNode class and complete the implementation of the abstract methods in that class. The tree search process needs to utilize a deep neural network. Given a state s, the DNN  estimates the state value of s as well as computes the policy at s (the probability of actions). You should implement such a neural network and use it in the implementation of some of the abstract methods.     

[Do not copy code from MCTSBase.py into your hw2.py file. Simply import the symbols from MCTSBase.]

## Submit your work
Put all your code in a single file named "hw2.py" (*you must use this file name*) and submit the file in moodle. 
(Different from hw1, you don't need to have code for downloading the weights of your trained model.)

We'll test your code as shown at the end of the gomoku.py file. The MCTS class will be imported from hw2.py. A MCTS object will be created and used in the NeuralMCTSPlayer to choose a move at each step. Make sure that your code works with gomoku.py.  
 
 
