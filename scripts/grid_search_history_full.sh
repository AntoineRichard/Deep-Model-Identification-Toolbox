# GRID SEARCH PARAMETERS
histories="1 2 3 4 5 6 8 10 12 14 16 18 20"

# VALIDATION PARAMETERS
run_max=9

# TRAINING PARAMETERS
iterations=80000
model="MLP_RELU_d32_d32"
lr=0.005
pred_points=1
batch_size=128
batch_val_size=10000
drop_rate=0.1
traj_length=20
mode="classic"

# PATH
root="/cs-share/dream/systemid_v2"
new_data_root="/cs-share/dream/kf_ws/src/lake_wanderer/records"
data_root="/cs-share/dream/systemid_v2/data"

training_set_1=("${data_root}/simu/train-full-random" "$new_data_root/GPS_PROCESSED_NH")
validation_set_1=("${data_root}/simu/val" "$new_data_root/FIX_VAL")
test_set_1=("${data_root}/simu/test" "$new_data_root/FIX_TEST")
DS_1_suffix=("perfect_simu_kf" "noisy_simu_kf")

training_set_2="${data_root}/real/kf_data_vrz_imu"
DS_2_suffix="real_kf"

training_set_3=("${data_root}/real/bebop" "${data_root}/simu/bebop")
DS_3_suffix=("real_bebop" "simu_bebop")
python_script="$root/networks/run_training.py"

# OUTPUT
output_root="/cs-share/dream/RSS-2020-Priorization"
to_file="$output_root/gridsearch_histories_full.txt"
save_suffix="$output_root/results/GS/HistorySearchFull"
tb_suffix="$output_root/tensorboard/GS/HistorySearchFull"

# INIT
mkdir -p ${output_root}

for (( j=0; j<${#training_set_1[*]}; j++ ))
do
	training_set=${training_set_1[$j]}
	validation_set=${validation_set_1[$j]}
	test_set=${test_set_1[$j]}
	ds_suffix=${DS_1_suffix[$j]}
	mkdir -p ${save_suffix}/${ds_suffix}
	for hist in ${histories}
	do
		for run in $(seq 0 $run_max)
		do
			mkdir -p ${save_suffix}/${ds_suffix}/${hist}/r${run}
			echo python3 ${python_script} --train_data ${training_set} --val_data ${validation_set} --test_data ${test_set} --batch_size ${batch_size} --val_batch_size ${batch_val_size} --input_dim 5 --output_dim 3 --dropout ${drop_rate} --model ${model} --learning_rate ${lr} --timestamp_idx 0 --output ${save_suffix}/${ds_suffix}/${hist}/r${run} --tb_dir ${tb_suffix}/${ds_suffix} --tb_log_name ${hist}-r${run} --reader_mode ${mode} --trajectory_length=${traj_length} --max_iterations ${iterations} --max_sequence_size ${hist}>> ${to_file}
		done
	done
done

ds_suffix=${DS_2_suffix}
mkdir -p ${save_suffix}/${ds_suffix}

echo ${save_suffix}/${ds_suffix}

for hist in ${histories}
do
	for run in $(seq 0 $run_max)
	do
		mkdir -p ${save_suffix}/${ds_suffix}/${hist}/r${run}
		echo python3 ${python_script} --train_data ${training_set} --val_data ${validation_set} --test_data ${test_set} --batch_size ${batch_size} --val_batch_size ${batch_val_size} --input_dim 5 --output_dim 3 --dropout ${drop_rate} --model ${model} --learning_rate ${lr} --timestamp_idx 0 --output ${save_suffix}/${ds_suffix}/${hist}/r${run} --tb_dir ${tb_suffix}/${ds_suffix} --tb_log_name ${hist}-r${run} --reader_mode ${mode} --trajectory_length=${traj_length} --max_iterations ${iterations} --max_sequence_size ${hist}>> ${to_file}
	done
done

for (( j=0; j<${#training_set_3[*]}; j++ ))
do
	training_set=${training_set_3[$j]}
	ds_suffix=${DS_3_suffix[$j]}
	mkdir -p ${save_suffix}/${ds_suffix}
	for hist in ${histories}
	do
		for run in $(seq 0 $run_max)
		do
			mkdir -p ${save_suffix}/${ds_suffix}/${hist}/r${run}
			echo python3 ${python_script} --train_data ${training_set} --val_ratio 0.1 --test_ratio 0.1 --batch_size ${batch_size} --val_batch_size 1000 --input_dim 8 --output_dim 4 --dropout ${drop_rate} --model ${model} --learning_rate ${lr} --output ${save_suffix}/${ds_suffix}/${hist}/r${run} --tb_dir ${tb_suffix}/${ds_suffix} --tb_log_name ${hist}-r${run} --reader_mode ${mode} --trajectory_length=${traj_length} --max_iterations ${iterations} --max_sequence_size ${hist}>> ${to_file}
		done
	done
done
