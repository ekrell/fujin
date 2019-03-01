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



