# Project1 实验报告

search

github链接:

## Question 1

本题要求实现深度优先搜索。深度优先搜索可通过递归实现，可借助一个栈来实现非递归的深度优先搜索。当需要进入到某个节点的子节点就把当前节点压入栈保存。又因为题目要求返回动作的列表，可以在每次进入到下一个节点的时候，把进入到这个节点的动作也压入栈中，最后在返回动作列表的时候，对这个栈进行一下处理换成列表即可。

注意到，对于每个节点，expand函数只能调用一次，在调用expand函数的时候，就算是对这个节点进行了遍历，而节点的孩子信息需要调用expand函数才能获得，而深度优先搜索有时候需要回溯，回溯到先前的节点的时候就不能再次调用expand函数获取孩子信息了。为解决这个问题，使用一个字典记录节点的孩子信息，这样回溯的时候就能够获知其孩子信息了。

代码及注释如下：

```python
def depthFirstSearch(problem):
    frontier = util.Stack() #用栈存节点
    pathSt = util.Stack() #存路径
    expanded = [] #记录已经扩展过的节点
    children_dict = {} #记录节点及其对应孩子列表
    frontier.push(problem.getStartState()) #把起点放进栈中
    while not frontier.isEmpty(): #当栈不空
        node = frontier.pop() #取出栈顶元素
        if problem.isGoalState(node): #判断是否目标状态，若是则返回路径
            path = []
            while not pathSt.isEmpty():
                path.insert(0,pathSt.pop()) #返回的是列表，需要处理一下栈中元素
            return path
        if node not in expanded: #检查节点是否扩展过，如果没扩展
            children = problem.expand(node)  #调用expand函数获取其孩子
            children_dict[node] = children #记录其孩子信息
        else:
            children = children_dict[node] #如果已经调用过expand，则从字典中取孩子信息
        if node not in expanded:
            expanded.append(node) 

        canExpand = 0 #一个标志，检查当前节点是否还能向深处扩展
        for nextState, action, cost in children: #对节点的每一个孩子
            if nextState not in expanded: #看看孩子是否有未扩展过的节点，若有
                frontier.push(node) #把当前节点入栈
                node = nextState #移动到这个孩子节点
                frontier.push(node) #这个孩子节点也入栈
                canExpand = 1 #修改标志
                pathSt.push(action) #加入动作
                break #跳出循环
        if not canExpand: #如果这个节点不能再向深处扩展
            pathSt.pop() #动作栈中弹出栈顶元素
    return []
```

## Question 2

本题要求实现广度优先搜索，广度优先搜索可借助一个先进先出队列来进行实现。但是由于util里面写好的Queue类没有很方便的检查一个元素是否在Queue中的接口，因此，这里用了一个list来记录进入队列的节点。由于广度优先搜索是按层遍历的，遍历节点的顺序并不是最终路径的顺序。在广度优先搜索中，扩展一个节点的时候，其父节点是确定的。因此，借助了两个字典，分别记录进入节点的父节点以及进入节点的动作，返回结果时可以方便的找到动作顺序。

代码及注释如下：

```python
def breadthFirstSearch(problem):
    expanded = [] #记录扩展过的节点
    recorded = [] #记录被添加到先进先出队列的节点
    stateFatherDic = {} #记录状态及其父节点，用于返回结果时得到动作路径
    stateActionDic = {} #记录状态及其对应的动作，用于返回结果时得到动作路径
    frontier = util.Queue() #先进先出的队列，按这个队列的顺序进行遍历
    frontier.push(problem.getStartState()) #将起点放到队列中
    recorded.append(problem.getStartState())
    while not frontier.isEmpty(): #当队列不空的时候
        node = frontier.pop() #取出队头元素
        if problem.isGoalState(node): #如果是目标状态，往回走得到动作路径
            path = []
            curNode = node
            while curNode in stateFatherDic:
                path.insert(0,stateActionDic[curNode])
                curNode = stateFatherDic[curNode]
            return path
        if node not in expanded: #如果节点没有扩展过
            expanded.append(node) #将节点加到扩展过的列表中
            children = problem.expand(node) #调用expand函数获取其孩子信息
            for nextState, action, cost in children: #对其每一个孩子
                if nextState not in recorded: #如果这个孩子没有被加入队列
                    stateFatherDic[nextState] = node #记录信息
                    stateActionDic[nextState] = action
                    recorded.append(nextState)
                    frontier.push(nextState) #将这个孩子加入队列
    return []
```



## Question 3

本题要求实现A\*搜索算法。A\*搜索算法接受一个启发式函数作为输入，它通过如下函数来计算每个节点的优先级
$$
f(n) = g(n) + h(n)
$$
其中f(n)是综合优先级，g(n)是节点n距离起点的代价，h(n)是节点n距离终点的预计代价，是由启发式函数确定的。在每次运算过程中，每次都从优先队列中选取综合优先级最小的节点作为下一个要遍历的节点。借助一个PriorityQueue来实现此算法。

代码及注释如下：

```python
def aStarSearch(problem, heuristic=nullHeuristic):
    nodeFatherDic = {} #记录进入节点的父节点
    nodeActionDic = {} #记录进入节点的动作
    #this cost not include cost from heuristic
    nodeMinCost = {} #节点距离起点的最小代价
    openSet = util.PriorityQueue() #优先队列，可选的遍历节点
    
    closeSet = [] #遍历过的节点
    openSet.push(problem.getStartState(),0) #将起点放入优先队列，并设代价为0
    nodeMinCost[problem.getStartState()] = 0
    while not openSet.isEmpty(): #当优先队列不空的时候
        node = openSet.pop() #取出代价最小的节点
        closeSet.append(node) #加入遍历过的节点集
        if problem.isGoalState(node): #检查是否目标状态，若是则处理一下动作序列并返回
            path = []
            curNode = node
            while curNode in nodeFatherDic:
                path.insert(0,nodeActionDic[curNode])
                curNode = nodeFatherDic[curNode]
            return path

        children = problem.expand(node) #调用expand函数获取其孩子信息
        for nextState, action, cost in children: #对其每一个孩子
            if nextState not in closeSet: #如果没有遍历过
                pastCost = nodeMinCost[node] + cost #计算此节点经其父节点到起点的代价 
                totalCost = nodeMinCost[node] + cost + heuristic(nextState,problem) #计算总和代价
                if nextState not in nodeMinCost or nodeMinCost[nextState] > pastCost: #如果距离起点代价更小，则代价信息
                    nodeMinCost[nextState] = pastCost
                    nodeFatherDic[nextState] = node #更新动作信息
                    nodeActionDic[nextState] = action
                    openSet.update(nextState, totalCost) #更新优先队列中的节点代价信息
                    
    return []
```



## Question 4

本题要求补充完成CornersProblem类，实现getStartState, isGoalState, expand, getNextState函数。在这个问题中，目标状态是搜索完四个角落。因此，除了当前位置以外，状态中需要记录搜索过的哪些角落以及角落的位置信息。

角落有四个，因此可以使用一个由0，1组成的四元组表示每一个角落是否被遍历过。

在getStartState函数中，返回的是当前位置，以及角落位置和对应的角落是否有被遍历过的四元组。

在isGoalState函数中，首先获取当前位置，然后检查当前位置是否在其中一个角落，若是，读取角落位置信息以及对应角落是否有被遍历过的四元组，因为元组不可更改，所以复制一份转成列表，修改列表中当前位于的角落为遍历过的状态。检查这个列表，看看是否四个角落都遍历过了，若是，则返回True，否则返回False

在getNextState函数中，输入当前状态和动作，返回下一个状态。首先获取当前位置、角落位置、遍历角落的情况。在这个函数中借助了Actions.directionToVector函数，再加上当前位置，能够得到下一个位置。检查下一个位置是否角落，如果是角落，修改遍历角落信息。将（下一个位置，（角落位置，角落遍历信息））作为下一个状态返回。

在expand函数中，

```python
class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and child function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"
        self.reachedCorners = (0,0,0,0)

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        if self.startingPosition in self.reachedCorners:
            self.reachedCorners[self.startingPosition] = 1
        return (self.startingPosition, (self.corners,self.reachedCorners))

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        curPos = state[0]
        current_reached_corners = list(state[1][1])
        corner_pos_list = list(self.corners)
        if curPos in self.corners:
            idx = corner_pos_list.index(curPos)
            current_reached_corners[idx] = 1
        for item in current_reached_corners:
            if item == 0:
                return False
        return True

    def expand(self, state):
        """
        Returns child states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (child,
            action, stepCost), where 'child' is a child to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that child
        """

        children = []
        for action in self.getActions(state):
            # Add a child state to the child list if the action is legal
            # You should call getActions, getActionCost, and getNextState.
            "*** YOUR CODE HERE ***"
            next_state = self.getNextState(state,action)
            children.append((next_state,action,1))

        curPos = state[0]
        corner_pos_list = list(self.corners)
        current_reached_corners = list(state[1][1])
        if curPos in self.corners:
            idx = corner_pos_list.index(curPos)
            current_reached_corners[idx] = 1
        self._expanded += 1 # DO NOT CHANGE
        return children

    def getActions(self, state):
        possible_directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        valid_actions_from_state = []
        for action in possible_directions:
            x, y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                valid_actions_from_state.append(action)
        return valid_actions_from_state

    def getActionCost(self, state, action, next_state):
        assert next_state == self.getNextState(state, action), (
            "Invalid next state passed to getActionCost().")
        return 1

    def getNextState(self, state, action):
        assert action in self.getActions(state), (
            "Invalid action passed to getActionCost().")
        x, y = state[0]
        dx, dy = Actions.directionToVector(action)
        nextx, nexty = int(x + dx), int(y + dy)
        "*** YOUR CODE HERE ***"
        current_reached_corners = list(state[1][1])
        if (nextx, nexty) in self.corners:
            idx = self.corners.index((nextx,nexty))
            current_reached_corners[idx] = 1
        return ((nextx, nexty), (self.corners,tuple(current_reached_corners)))

    def getCostOfActionSequence(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)

```



## Question 5

本题要求实现一个启发式函数，针对遍历四个角落的问题来设计。由于启发式函数估计的是节点到终点的代价，而本题的目标状态是需要遍历完四个节点。启发式函数在估计代价的时候除了要知道当前位置以外，还需要知道当前状态下哪些角落已经遍历过了，哪些还没有，以及四个角落的位置信息。



## Question 6

## Question 7

## 附录

### 题目完成得分

![image-20211011195123868](C:\Users\59680\AppData\Roaming\Typora\typora-user-images\image-20211011195123868.png)

![image-20211011195139801](C:\Users\59680\AppData\Roaming\Typora\typora-user-images\image-20211011195139801.png)

![image-20211011195157253](C:\Users\59680\AppData\Roaming\Typora\typora-user-images\image-20211011195157253.png)

![image-20211011195218359](C:\Users\59680\AppData\Roaming\Typora\typora-user-images\image-20211011195218359.png)

![image-20211011195243034](C:\Users\59680\AppData\Roaming\Typora\typora-user-images\image-20211011195243034.png)

![image-20211011195322061](C:\Users\59680\AppData\Roaming\Typora\typora-user-images\image-20211011195322061.png)

![image-20211011195331872](C:\Users\59680\AppData\Roaming\Typora\typora-user-images\image-20211011195331872.png)

![image-20211011195414946](C:\Users\59680\AppData\Roaming\Typora\typora-user-images\image-20211011195414946.png)

![image-20211011195450015](C:\Users\59680\AppData\Roaming\Typora\typora-user-images\image-20211011195450015.png)

![image-20211011195055152](C:\Users\59680\AppData\Roaming\Typora\typora-user-images\image-20211011195055152.png)



### 完整代码文件

