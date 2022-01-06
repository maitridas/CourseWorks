from math import fabs
from itertools import combinations
from astar import Astar
from copy import deepcopy
import yaml

class Location:
    def __init__(self,x=-1,y=-1):
        self.x = x
        self.y = y

    def __eq__(self,other):
        return ((self.x==other.x) and (self.y == other.y))

class Robot:
    def __init__(self,time,location):
        self.time = time
        self.location = location
    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def is_equal_except_time(self, state):
        return self.location == state.location
    def __hash__(self):
        return hash(str(self.time)+str(self.location.x) + str(self.location.y))

class Task:
    def __init__(self,location):
        self.location = location
    def is_equal(self, state):
        return self.location == state.location

class VertexConstraint:
    def __init__(self, time, location):
        self.time = time
        self.location = location
    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash(str(self.time)+str(self.location))

class Constraints(object):
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

# to be reviewed
class Conflict(object):
    VERTEX = 1
    EDGE = 2
    def __init__(self):
        self.time = -1
        self.type = -1

        self.agent_1 = ''
        self.agent_2 = ''

        self.location_1 = Location()
        self.location_2 = Location()

#to be reviwed
class EdgeConstraint(object):
    def __init__(self, time, location_1, location_2):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2
    def __eq__(self, other):
        return self.time == other.time and self.location_1 == other.location_1 \
            and self.location_2 == other.location_2

class Grid:
    def __init__(self,dimension,robots,tasks,obstacles,temporary_locations):
        self.dimension = dimension
        self.robots = robots
        self.tasks = tasks 
        self.obstacles = obstacles
        self.temporary_locations = temporary_locations

        self.all_robots = {}
        self.make_Robot_dict()#to be implemented

        self.all_tasks = {}
        self.make_task_dict()#to be implemented

        self.constraints = Constraints()
        self.constraint_dict = {}

        self.a_star = Astar(self)

        self.tasks_assigned = {}

    def heuristics(self,robot,state,goal):
        return fabs(state.location.x - goal.location.x) + fabs(state.location.y - goal.location.y)

    def is_at_goal(self,robot,current,goal):
        return current.location == goal.location#change something check
    
    def get_neighbors(self, robot):
        neighbors = []

        #Action: Wait for 1 time
        n = Robot(robot.time + 1, robot.location)
        if self.robot_state_valid(n):
            neighbors.append(n)

        #Action:Move left
        n = Robot(robot.time +1, Location(robot.location.x - 1,robot.location.y))
        if self.robot_state_valid(n):
            neighbors.append(n)
        
        #Action:Move Right
        n = Robot(robot.time +1, Location(robot.location.x + 1,robot.location.y))
        if self.robot_state_valid(n):
            neighbors.append(n)
        
        #Action:Move Up
        n = Robot(robot.time +1, Location(robot.location.x ,robot.location.y + 1))
        if self.robot_state_valid(n):
            neighbors.append(n)

        #Action:Move Down
        n = Robot(robot.time +1, Location(robot.location.x ,robot.location.y - 1))
        if self.robot_state_valid(n):
            neighbors.append(n)

        return neighbors
        

    def robot_state_valid(self, robot):
        value = True
        for elem in self.constraints.vertex_constraints:
            if elem == VertexConstraint(robot.time, robot.location):
                value = False
                break

        return (robot.location.x >= 0 and robot.location.x < self.dimension[0] 
                and robot.location.y >= 0 and robot.location.y < self.dimension[1] 
                and value #expression not working as expected
                and (robot.location.x, robot.location.y) not in self.obstacles)

    def assign_task(self,robot,current):
        """
            assigns the nearest task to the robot from its current location
            it uses the a different heuristics and adds the f cost 
        """
        min_value = float('inf')
        final_task = Task(Location(-1,-1))
        for task in self.all_tasks:
            if task not in self.tasks_assigned.values():
                start = self.all_tasks[task]["start"]
                curr_value = self.task_assign_function_value(robot,current,start)
                if curr_value < min_value:
                    final_task = task
                    min_value = curr_value
                
        self.tasks_assigned[robot] = final_task
        print("assigned task" + final_task)
        return final_task

    def task_assign_function_value(self,robot,current,task):
        start = self.all_robots[robot]["start"]
        return (fabs(start.location.x-current.location.x)+
                fabs(start.location.y-current.location.y)+
                fabs(current.location.x-task.location.x)+
                fabs(current.location.y-task.location.y))

    def find_soln(self):
        soln = {}#needs to update assigned task
        self.tasks_assigned = {}
        for robot in self.all_robots.keys():
            self.constraints = self.constraint_dict.setdefault(robot, Constraints())
            simple_soln = []
            simple_soln = self.a_star.search_for_path(robot) # changed something check it out
            if not simple_soln:
                return False
            soln.update({robot:simple_soln})
        return soln

    def find_soln_cost(self,soln):
        return sum([len(path) for path in soln.values()])

    # to be changed
    def get_constraint_dict(self, solution):
        max_t = max([len(plan) for plan in solution.values()])
        result = Conflict()
        temp_dict = {}
        for t in range(max_t):
            constraint_dict = {}
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                #returns the cobination of all possible robots combination to detect collisions
                state_1 = self.get_state(agent_1, solution, t)
                state_2 = self.get_state(agent_2, solution, t)
                if state_1.is_equal_except_time(state_2):
                    result.time = t
                    result.type = Conflict.VERTEX
                    result.location_1 = state_1.location
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2

                    if (state_1.time == t) and (state_2.time == t):
                        v_constraint = VertexConstraint(result.time, result.location_1)
                        constraint = Constraints()
                        constraint.vertex_constraints |= {v_constraint}
                        constraint_dict[result.agent_1] = constraint
                        constraint_dict[result.agent_2] = constraint
                        temp_dict |= constraint_dict
                        return temp_dict
                    else:
                        v_constraint = VertexConstraint(state_1.time, result.location_1)
                        constraint = Constraints()
                        constraint.vertex_constraints |= {v_constraint}
                        constraint_dict[result.agent_1] = constraint

                        v_constraint = VertexConstraint(state_2.time, result.location_1)
                        constraint = Constraints()
                        constraint.vertex_constraints |= {v_constraint}
                        constraint_dict[result.agent_2] = constraint
                    
                        temp_dict |= constraint_dict
                        return temp_dict

            constraint_dict = {}
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_1b = self.get_state(agent_1, solution, t+1)

                state_2a = self.get_state(agent_2, solution, t)
                state_2b = self.get_state(agent_2, solution, t+1)

                if state_1a.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2a):
                    result.time = t
                    result.type = Conflict.EDGE
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    result.location_1 = state_1a.location
                    result.location_2 = state_1b.location

                    constraint1 = Constraints()
                    constraint2 = Constraints()

                    e_constraint1 = EdgeConstraint(result.time, result.location_1, result.location_2)
                    e_constraint2 = EdgeConstraint(result.time, result.location_2, result.location_1)

                    constraint1.edge_constraints |= {e_constraint1}
                    constraint2.edge_constraints |= {e_constraint2}

                    constraint_dict[result.agent_1] = constraint1
                    constraint_dict[result.agent_2] = constraint2
                    temp_dict |= constraint_dict
                    return temp_dict

        return False

    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    def make_Robot_dict(self):
        for agent in self.robots:
            start_state = Robot(0, Location(agent['start'][0], agent['start'][1]))
            goal_state = Robot(0, Location(agent['goal'][0], agent['goal'][1]))

            self.all_robots.update({agent['name']:{'start':start_state, 'goal':goal_state}})

    def make_task_dict(self):
        for task in self.tasks:
            start_state = Task(Location(task['start'][0], task['start'][1]))
            goal_state = Task(Location(task['goal'][0], task['goal'][1]))

            self.all_tasks.update({task['name']:{'start':start_state, 'goal':goal_state}})
    

class HighLevelNode:
    def __init__(self):
        self.soln = {}
        self.constraint_dict = {}
        self.cost = 0

    def __lt__(self, other):
        return self.cost < other.cost

class CBS:
    def __init__(self, grid):
        self.grid = grid
        self.frontier = set()
        self.explored = set()

    def search(self):
        start = HighLevelNode()
        start.constraint_dict = {}
        for robot in self.grid.all_robots.keys():
            start.constraint_dict[robot] = Constraints()
        
        start.soln = self.grid.find_soln()

        if not start.soln:
            return {}

        start.cost = self.grid.find_soln_cost(start.soln)

        self.frontier |= {start}

        while self.frontier:
            node = min(self.frontier)
            self.frontier -= {node}
            self.explored |= {node}

            self.grid.constraint_dict = node.constraint_dict
            conflict_dict = self.grid.get_constraint_dict(node.soln)#to be implemented

            if not conflict_dict:
                print("solution found")

                return self.generate_plan(node.soln)

            constraint_dict = conflict_dict 

            for agent in constraint_dict.keys():
                new_node = deepcopy(node)
                new_node.constraint_dict[agent].add_constraint(constraint_dict[agent])

                self.grid.constraint_dict = new_node.constraint_dict
                new_node.solution = self.grid.find_soln()
                if not new_node.solution:
                    continue
                new_node.cost = self.grid.find_soln_cost(new_node.solution)

                conflict_dict = self.grid.get_constraint_dict(node.soln)#to be implemented

                if not conflict_dict:
                    print("solution found")
                    return {}
                    
                
                if new_node not in self.explored:
                    self.frontier |= {new_node}

        return {}

    def generate_plan(self,soln):
        plan = {}
        for agent, path in soln.items():
            path_dict_list = [{'t':state.time, 'x':state.location.x, 'y':state.location.y} for state in path]
            plan[agent] = path_dict_list
        return plan

def main():

    with open("input_data.yaml", 'r') as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    robots = param['robots']
    tasks = param["task"]
    temporary_locations = param["map"]["temporary"]    

    grid = Grid(dimension,robots,tasks,obstacles,temporary_locations)  

    cbs = CBS(grid)
    solution = cbs.search()
    if not solution:
        print(" Solution not found" )
        return

    # Write to output file
    with open("schedule.yaml", 'r') as output_yaml:
        try:
            output = yaml.load(output_yaml, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    output["schedule"] = solution
    output["cost"] = grid.find_soln_cost(solution)
    with open("schedule.yaml", 'w') as output_yaml:
        yaml.safe_dump(output, output_yaml)

if __name__ == "__main__":
    main()