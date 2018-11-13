## fujin

A game-theoretic path planner. 

### Based on

    Robot motion planning: A game-theoretic foundation. S. M. LaValle. Algorithmica, 26(3):430--465, 2000.

### Examples

__Example A: a single environment vector acts against the traveler__

	python basic_planner.py -o test/data/EXP1_region.png  \    # Region 
                            -u test/data/EXP1_water_u.png \    # Vector field, u components
                            -v test/data/EXP1_water_v.png \    # Vector field, v components
                            -w 1 \                             # Weight of vector field
                            -e .5 \                            # Forecast/measurement error
                            -s 0,0 \                           # Traveler start coordinates
                            -t 250,250 \                       # Traveler goal coordinates
                            -n EXP1_nash.txt \                 # File to save nash solution grid
                            -c EXP1_cost2go.txt                # File to save cost2go solution grid

__Example B: two vectors, wind and water act against the traveler__

    python basic_planner.py -o test/data/EXP1_region.png \
                            -u test/data/EXP1_water_u.png,test/data/EXP1_wind_u.png \
                            -v test/data/EXP1_water_v.png,test/data/EXP1_wind_v.png \
                            -w 0.25,0.75 \ 
                            -e .5,0.15 \
                            -s 0,0 \
                            -t 250,250 \
                            -n EXP1_nash2.txt \
                            -c EXP1_cost2go.txt

