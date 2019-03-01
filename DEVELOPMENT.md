# Information for developers

## Important Data Structures


### Dictionaries

The code makes use of several dictionaries that appear throughout.
The expected key:value contents are described below.  

**environment**
Stores a variety of data elements that describe the region, environment
in which motion planning is taking place. Sometimes abbreviated env. 

__Key:Value Pairs__
- iterations (int): Number of times to run dynamic programming solver.
- speed (float): Traveler's speed in cells per second.
- occupancy (List of str): Paths to occupancy grids.
- ucomponents (List of str): Paths to u component grids.
- vcomponents (List of str): Paths to v component grids.
- weights (List of floats): Constant weights per force.
- weightgrids (List of str): Paths to weight grids per force.
- errors (List of floats): Constant errors per force.
- errorgrids (List of str): Paths to error grids per force.
- start (tuple): tuple containing:
    - (int) Row of traveler's start location.
    - (int) Column of traveler's start location.
- target (tuple): tuple containing:
    - (int) Row of traveler's target location.
    - (int) Column of traveler's target location.
- files (dict): A dict with the following keys:
    - cost2go: Path to ASCII cost2go grid.
    - work2go: Path to ASCII work2go grid.
    - actiongrid: Path to ASCII work2go grid.
    - pickle: Path to pickle file for results archive.
    - pandas: Path to results stats csv.
    - plots: Path (prefix) for result figures.
- verbose (bool): Whether to print verbose output.
- reuse (bool): Whether to reuse motion plan to get stats, figures.
- bounds (dict): A dict with the following keys:
    - upperleft (tuple): A tuple containing:
        - (int): Row of upper-left corner of boundary to solve.
        - (int): Column of upper-left corner of boundary to solve.
    - lowerright (tuple): A tuple containing:
        - (int): Row of lower-right corner of boundary to solve.
        - (int): Column of lower-right corner of boundary to solve.


__Related functions__
- env_setup.py -> parseOptions : Creates new environment using command line args. 

**traveler**
Represents the agent who is trying to generate a motion plan to reach a goal position. 

__Key:Value Pairs__
- start (int, int): row, col coordinates of start position.
- target (int, int): row, col coordinates of target position.
- actionspace (list of strings): Possible movements as symbols (ex: "^" -> up).
- action2radians (Dict of string:float): Movement symbols query movement angle (ex: "^" -> (pi/2)).
- speed_cps (float): Speed of traveler in grid cells per second.

__Related functions__
- env_setup.py -> getTraveler : Creates a new traveler. 

**pstat**
Holds statistics of following a particular action sequence (path)
in a given cost2go, work2go. 

__Key:Value Pairs__
- distances (List of float): Distances of path.
- distance_sum (float): Total distance of path.
- num_cells (int): Number of cells path visits.
- num_waypoints (int): Number of path waypoints.
- costs (List of float): Cost of each action.
- cost_sum (float): Total cost of path.
- works (List of float): Work for each action.
- work_sum (float): Total work of path.

__Related functions__
- travel_tools -> statPath : Creates a new pstat.



