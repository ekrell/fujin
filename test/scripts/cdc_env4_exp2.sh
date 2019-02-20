EXP=CDC_ENV4_EXP2
python basic_planner.py --verbose -o test/data/env_4.tif  -s 0,0 -t 29,29 -i 10 \
    -c test/results/$EXP""_cost2go.txt -a test/results/$EXP""_actions.txt -x test/results/$EXP""_work2go.txt \
    --pandas test/results/$EXP""_history.pandas --plots test/results/$EXP""_history_plots \
    -u test/data/env_4_u_exp2.tif -v test/data/env_4_v_exp2.tif  -w 1 -e 25
