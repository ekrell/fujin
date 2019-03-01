# Information for developers

## Important Data Structures


### Dictionaries

The code makes use of several dictionaries that appear throughout.
The expected key:value contents are described below.  

**environment**
Stores a variety of data elements that describe the region, environment
in which motion planning is taking place. Sometimes abbreviated env. 

**traveler**
Represents the agent who is trying to generate a motion plan to 
reach a goal position. 

- start (int, int): row, col coordinates of start position.
- target (int, int): row, col coordinates of target position.
- actionspace (list of strings): Possible movements as symbols (ex: "^" -> up).
- action2radians (Dict of string:float): Movement symbols query movement angle (ex: "^" -> (pi/2)).
- speed_cps (float): Speed of traveler in grid cells per second.

__Related functions:__
- env_setup.py -> getTraveler : Creates a new traveler. 

**pstat**
Holds statistics of following a particular action sequence (path)
in a given cost2go, work2go. 

- distances (List of float): Distances of path.
- distance_sum (float): Total distance of path.
- num_cells (int): Number of cells path visits.
- num_waypoints (int): Number of path waypoints.
- costs (List of float): Cost of each action.
- cost_sum (float): Total cost of path.
- works (List of float): Work for each action.
- work_sum (float): Total work of path.

__Related functions:__
- travel_tools -> statPath : Creates a new pstat.



