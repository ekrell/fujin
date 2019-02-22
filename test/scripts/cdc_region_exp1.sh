# For CDC 2019 submission
# In paper: Region 3, Experiment

EXP=CDC_REG_EXP1

python basic_planner.py --speed 10 -o test/data/region.tif -u test/data/water_u_100x.tif -v test/data/water_v_100x.tif -w 1 -s 850,600 -t 500,550  -c test/results/$EXP""_cost2go.txt -a test/results/$EXP""_actions.txt -x test/results/$EXP""_work2go.txt -b 400,400,900,800 --verbose --pickle test/results/$EXP""_history.pickle --pandas test/results/$EXP""_history.pandas --plots test/results/$EXP""_history_plots -i 1
