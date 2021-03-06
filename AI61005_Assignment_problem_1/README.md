# AI61005_Assignment_problem_1
# Target Assignment and Path Finding

## Introduction
This repository consists of python code solving problem 1 of Assignment 1

## Dependencies
Install the necessary depandencies by running

```shell
pip3 install -r requirements.txt
```

## Execution
Before executing the line below update the input_data.yaml file in the format to take input

``` 
python grid.py 
```
Update the input_data.yaml file in the format and output will be provided in schedule.yaml file

## Algorithm Used (REPORT)
### Conflict Based Search(CBS)
Asigned the closest task to a robot and found the solution with the low level CBS( that's based on A* algorithm), then found conflict and formed nodes further to find the optimal path and task allocation for all robots using the High-level search

#### Psuedo code for CBS

High Level search

```
Input: grid instance
R.constraints = {}
R.solution = find individual paths using the low level()
R.cost = cost_function(R.solution)
insert R to frontier
while frontier not empty do 
    node <- minimun cost node from the frontier
    Validate the paths in node until a conflict occurs.
    if node has no conflict then
        return node.solution//node id the goal

    C <- first conflict(robot_1,robot_2, time) in node
    for each robot in C do
        A <- new node 
        A.constraints <- node.constraints
        A.sloution <- node.solution
        Update A.solution by invoking low-level(robot)
        A.cost = cost_function(A.solution)
        Insert A to frontier

```

Low Level search is based on A* finds the simplest solution for all robots
```
 make an frontier containing only the starting node
   make an empty explored\\(to keep track of all explored nodes)
   while (the destination node has not been reached):
       consider the node with the lowest f score in the frontier
       if (this node is our destination node) :
           we are finished 
       if not:
           put the current node in the explored and look at all of its neighbors
           for (each neighbor of the current node):
               if (neighbor has lower g value than current and is in the explored) :
                   replace the neighbor with the new, lower, g value 
                   current node is now the neighbor's parent            
               else if (current g value is lower and this neighbor is in the frontier ) :
                   replace the neighbor with the new, lower, g value 
                   change the neighbor's parent to our current node

               else if this neighbor is not in both lists:
                   add it to the frontier and set its g
```

### Heuristics Used
Heuristic used is based on the manhattan distance from a particuler node
```
heuristic_value = |current_x_coordinate - goal_x_coordinate| + |current_y_coordinate - goal_y_coordinate|
```
Tasks are also assigned using the same logic 

| | -> mod function

### Task Assignment
Task is assigned using a heuristics which takes into account the manhattan distance between the task and robot and assigns the shortest distance task to the robot
```
task_assignment_value = |robot_start_x_coordinate - task_pick_up_x_coordinate| + |robot_start_y_coordinate - task_pick_up_y_coordinate|
```
### Searched in segments
A* algorithm(low level search) is runned in segments like from robot start to task pickup then from task pickup to task drop off and then from task drop off to robot destination

## Results
For a 1 Robot system -
input and output is given in the following respective files above - input_data.yaml and schedule.yaml

### Algorithm and result is written more clearly in the report