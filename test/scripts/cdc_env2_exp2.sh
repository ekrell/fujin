# For CDC 2019 submission
# In paper: Region 1, Experiment 2

# Forces against vehicle

EXP=CDC_ENV2_EXP2
python basic_planner.py --speed 10 --verbose -o test/data/env_2.tif  -s 9,2 -t 1,2 -i 10 \
    -c test/results/$EXP""_cost2go.txt -a test/results/$EXP""_actions.txt -x test/results/$EXP""_work2go.txt \
    --pandas test/results/$EXP""_history.pandas --plots test/results/$EXP""_history_plots \
    -u test/data/env_2_u_exp2.tif -v test/data/env_2_v_exp2.tif  -w 1 -e 0
