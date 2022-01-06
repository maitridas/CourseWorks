
class Astar:
    def __init__(self,grid):
        self.all_robots = grid.all_robots
        self.all_tasks = grid.all_tasks
        self.heuristics = grid.heuristics
        self.is_at_goal = grid.is_at_goal
        self.get_neighbors = grid.get_neighbors
        #self.assigned_task = grid.assigned_task
        self.assign_task = grid.assign_task
        self.total_path = []
        self.max_path_length = 14
        self.robot_state_valid = grid.robot_state_valid

    def return_path(self,current,path_dict):
        path = [current]
        while current in path_dict.keys():
            current = path_dict[current]
            path.append(current)
        return path[::-1]#reverses the path

    def search_in_segments(self,robot,start,goal):
        """
        low level search
        searches a simple path for the agent 
        """
        cost = 1
        
        #initialize frontier and explored to start
        frontier = {start}
        explored = set()

        #initialize path_dict : used to keep track of path taken
        path_dict = {}

        #intialize the g ie..., g(n)
        g_dict = {}
        g_dict[start] = 0

        #initialize function score i.e...., f(n)=g(n) + h(n)
        f_dict = {}
        f_dict[start] = self.heuristics(robot,start,goal)#TODO

        # Run the loop till there is something in the frontier
        while frontier:
            temp = {element:f_dict.setdefault(element,float("inf")) for element in frontier}
            current = min(temp, key=temp.get)

            if self.is_at_goal(robot,current,goal):
                return self.return_path( current, path_dict )

            frontier -= {current}
            explored |= {current}

            neighbors = self.get_neighbors(current)# check once later some problem here

            for neighbor in neighbors:
                if neighbor in explored:
                    continue
                #needs checking
                g_value = g_dict.setdefault(current,float("inf")) + cost

                if neighbor not in frontier:
                    frontier |= {neighbor}
                elif g_value >= g_dict.setdefault(neighbor, float("inf")):
                    continue

                path_dict[neighbor] = current

                g_dict[neighbor] = g_value
                f_dict[neighbor] = g_dict[neighbor] + self.heuristics(robot,neighbor,goal)#TODO

        return False          

    def insert_to_total_path(self,path):
        for node in path:
            self.total_path.append(node)

    def search_for_path(self,robot):
        self.total_path = []
        start = self.all_robots[robot]["start"]
        task = self.assign_task(robot,start)
        task_start = self.all_tasks[task]["start"]
        path = self.search_in_segments(robot,start,task_start)
        self.insert_to_total_path(path)
        
        start = self.total_path[-1]
        task_final = self.all_tasks[task]["goal"]
        path = self.search_in_segments(robot,start,task_final)
        path.pop(0)
        self.insert_to_total_path(path)
        
        start = self.total_path[-1]
        robot_final = self.all_robots[robot]["goal"]
        path = self.search_in_segments(robot,start,robot_final)
        path.pop(0)
        self.insert_to_total_path(path)

        return self.total_path


