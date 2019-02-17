EXP=CDC_ENV2_TEST
python basic_planner.py --verbose -o test/data/env_2.tif  -s 9,2 -t 1,2 -i 10 \ #-b 0,0,9,9 \
    -c test/results/$EXP""_cost2go.txt -a test/results/$EXP""_actions.txt -x test/results/$EXP""_work2go.txt \
    --pandas test/results/$EXP""_history.pandas --plots test/results/$EXP""_history_plots
