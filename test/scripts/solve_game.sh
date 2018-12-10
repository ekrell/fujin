G=game_4way_res50_scale100

R=(201 333 466 599)
C=(201 333 466 599)
I=(10 20 30 40 50 60 70 80 90 100)

for r in "${R[@]}"
do
    for c in "${C[@]}"
    do
        GAME=$G""_$r""-$c""
        for i in "${I[@]}"
        do
            python ../../solve_game.py -g $GAME"".txt -i $i >> results_$GAME"".txt
       done
    done
done

