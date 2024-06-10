model_type_list='f_xf ann_d_512 ann_d_1024 lstm_l_1 lstm_l_2 lstm_l_3 cnn'
for model_type in $model_type_list; do
    echo $model_type
    # Run
    python build_and_run.py dc 0 $model_type dc
done