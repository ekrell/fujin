## fujin

### Example

	python basic_planner.py -o test/data/EXP1_region.png  \    # Region 
                            -u test/data/EXP1_water_u.png \    # Vector field, u components
                            -v test/data/EXP1_water_v.png \    # Vector field, v components
                            -w 1 \                             # Weight of vector field
                            -e .5 \                            # Forecast/measurement error
                            -s 0,0 \                           # Traveler start coordinates
                            -t 250,250 \                       # Traveler goal coordinates
                            -n EXP1_nash.txt \                 # File to save nash solution grid
                            -c EXP1_cost2go.txt                # File to save cost2go solution grid


