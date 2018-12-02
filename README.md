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
                            -c EXP1_cost2go.txt \              # File to save cost2go solution grid
                            -x EXP1_work2go.txt \              # File to save applied work grid
                            -a EXP1_actions.txt                # File to save action grid

__Example B: two vectors, wind and water act against the traveler__

    python basic_planner.py -o test/data/EXP1_region.png \
                            -u test/data/EXP1_water_u.png,test/data/EXP1_wind_u.png \
                            -v test/data/EXP1_water_v.png,test/data/EXP1_wind_v.png \
                            -w 0.25,0.75 \ 
                            -e .5,0.15 \
                            -s 0,0 \
                            -t 250,250 \
                            -c EXP2_cost2go.txt \
                            -x EXP2_work2go.txt \
                            -a EXP2_actions.txt

__Example C: Same as example A, but reusing solutions for new startpoint__

	python basic_planner.py -o test/data/EXP1_region.png  \    # Region 
                            -u test/data/EXP1_water_u.png \    # Vector field, u components
                            -v test/data/EXP1_water_v.png \    # Vector field, v components
                            -w 1 \                             # Weight of vector field
                            -e .5 \                            # Forecast/measurement error
                            -s 150,150 \                       # Traveler start coordinates
                            -t 250,250 \                       # Traveler goal coordinates
                            -c EXP3_cost2go.txt \              # File to save cost2go solution grid
                            -x EXP3_work2go.txt \              # File to save applied work grid
                            -a EXP3_actions.txt                # File to save action grid
                            -r                                 # Reuse solution

__Example D: Using geotiffs__

	python basic_planner.py -o test/data/EXP2_region.tif  \    # Region 
                            -u test/data/EXP2_water_u.tif \    # Vector field, u components
                            -v test/data/EXP2_water_v.tif \    # Vector field, v components
                            -w 1 \                             # Weight of vector field
                            -e .5 \                            # Forecast/measurement error
                            -s 150,150 \                       # Traveler start coordinates
                            -t 250,250 \                       # Traveler goal coordinates
                            -c EXP4_cost2go.txt \              # File to save cost2go solution grid
                            -x EXP4_work2go.txt \              # File to save applied work grid
                            -a EXP4_actions.txt                # File to save action grid




### Todo

- [X] Accept geotiff
- [X] Accept grid of errors
- [X] Accept grid of weights
- [X] Measure solver results
- [X] Track convergence
- [X] Pandas convergence
- [X] Graph convergence
- [ ] Compare to: no uncertainty
- [ ] Compare to: no work
- [ ] Compare to: metahueristic planner
- [X] Decide output format
- [X] Convert to waypoints
- [X] Specify subregion to solve
- [ ] 8 direction movement
- [X] Save convergence history as pickle
