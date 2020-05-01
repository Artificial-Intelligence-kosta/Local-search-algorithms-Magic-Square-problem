# -*- coding: utf-8 -*-
"""
Magic Square class
"""
import numpy as np
class MagicSquare(object):
    def __init__(self,n):
        """
        Create Magic Square matrix with shape=(n,n),
        where all elements are different and in range from 1 to n^2.
        Elements are sorted in ascending order
        
        Parameters:
        -----
        n: dimension of the square
        """
        self.n = n
        elements = range(1, n**2+1)
        self.M = np.array(elements,dtype = np.int32).reshape((n,n))
    def genRandSquare(self):
        """
        Generates random Magic Square matrix.
        """
        self.M = np.random.permutation(range(1,self.n**2+1)).reshape(self.n,self.n)
    def getNumOfViolated(self):
        """
        Gets the number of violated constraints. Constraints are:
        Sum of elements along each row, each column, main diagonal
        and non-main diagonal should be n*(n^2+1)/2
        
        Returns:
        -----
        Integer that represents sum of violated constraints
        """
        target_sum = self.n*(self.n**2 + 1)/2 #sum that should be satisfied
        num_of_violated = 0
        #check rows
        rows_sum = np.sum(self.M, axis=1) # get sum for each row
        num_of_violated += np.sum(rows_sum != target_sum)
        #check columns
        columns_sum = np.sum(self.M, axis=0) # get sum for each column 
        num_of_violated += np.sum(columns_sum != target_sum)
        #check main diagonal
        num_of_violated += (np.trace(self.M) != target_sum)
        #check non-main diagonal
        num_of_violated += (np.trace(np.flip(self.M,axis=0)) != target_sum)
        return num_of_violated
    def getSuccessors(self,mode,k=None):
        """
        Get successors for the current state of the Magic Square.
        Possible actions are flippings of the any 2 elements in the square.
        
        Parameters: 
        -----
        mode: string that indicates whether you want top k, or random k successors
        ("top" for top k, "random" for random k)
        k: number of successors you want to get.If k is None get all successors (nC2),
        they will be sorted in ascending order.
        
        Returns:
        -----
        list of successors. In the "top" mode list is sorted, in the ascending 
        order, by the number of violated constraints.
        """
        successors = []
        if k == None: # return all successsors: nC2
            k = int(self.n**2*(self.n**2-1)/2)
            mode = 'top'
        # if you want random k successors
        if (mode == 'random'):
            i = 0
            while i < k:
                # generate 4 radnom indicies for swapping
                #x1,x2,y1,y2 = np.random.choice(range(self.n), size=4, replace=True)
                x1,x2,y1,y2 = np.random.randint(low=0,high=self.n,size=4)
                if (x1==x2) and (y1==y2):
                    continue
                successor = self.getSuccessor(x1,y1,x2,y2)
                exist = False
                for s in successors: # check if this successor allready exists in the list
                    if np.array_equal(s.M,successor.M):
                        exist = True
                if not exist:
                    successors.append(successor)
                    i += 1
                
        # if you want top k successors
        elif (mode == 'top'):
            for x1 in range(self.n):
                for y1 in range(self.n):
                    for x2 in range(x1,self.n):
                        for y2 in range(self.n):   
                            if (x1 == x2) and (y2<=y1):
                                continue
                            successor = self.getSuccessor(x1,y1,x2,y2)
                            # place successor in final list, only if it is in top k
                            if not successors: # if there are no successors, append current
                                successors.append(successor)
                            else:
                                isPlaced = False # flags if you placed the successor in the list or not
                                for i in range(len(successors)): # place the successor on the right place in the list
                                    if (successor.getNumOfViolated() < successors[i].getNumOfViolated()):
                                        successors.insert(i,successor)
                                        isPlaced = True
                                        break
                                if (len(successors)>=k): # if there are more than k successors delete excess
                                    successors = successors[:k]
                                else: # if there are no k successors in the list and you have not placed successor, add it to the list
                                    if not isPlaced:
                                        successors.append(successor)
        else:
            raise ValueError("The mode argument is wrong! Take a look in the decription.")
        return successors
            
    def getSuccessor(self,x1,y1,x2,y2):
        successor = MagicSquare(self.n)
        successor.M = self.M.copy()
        # swap values
        successor.M[x1,y1],successor.M[x2,y2] = successor.M[x2,y2],successor.M[x1,y1]
        return successor
    def printSquare(self):
        """
        Prints out the Magic Square
        """
        print("Square: ")
        print(self.M)
    
        
        
        
        
    

