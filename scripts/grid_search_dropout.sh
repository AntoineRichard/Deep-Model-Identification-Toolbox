# GRID SEARCH PARAMETERS
learning_rates_list="0.0005 0.001 0.0025 0.005"
drop_rates=" 0.0 0.025 0.05 0.075 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5"
architectures_list_b=("MLP_" "ATTNMPMH_" "CNN_" "RNN_" "GRU_" "LSTM_")
architectures_list_e=("_d32_r_d32_r" "_a1.75_md32_d32_r_d32_r" "_k3c12_k3c12_d32_r" "_hs32_l2_d32_r_d32_r" "_hs32_l2_d32_r_d32_r" "_hs32_l2_d32_r_d32_r")
mode_list=("classic" "seq2seq" "classic" "seq2seq" "seq2seq" "seq2seq")

# VALIDATION PARAMETERS
run_max=4

# TRAINING PARAMETERS
iterations=100000
act="RELU"
used_points=6
pred_points=1
batch_size=128
batch_val_size=10000
drop_rate=0.1
traj_length=20

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
to_file="$output_root/gridsearch_dropout.txt"
save_suffix="$output_root/results/GS/DropoutSearch"
tb_suffix="$output_root/tensorboard/GS/DropoutSearch"

# INIT
mkdir -p ${output_root}

for (( j=0; j<${#training_set_1[*]}; j++ ))
do
	training_set=${training_set_1[$j]}
	validation_set=${validation_set_1[$j]}
	test_set=${test_set_1[$j]}
	ds_suffix=${DS_1_suffix[$j]}
	mkdir -p ${save_suffix}/${ds_suffix}
	for (( i=0; i<${#architectures_list_b[*]}; i++ ))
	do
		model_suffix=${ds_suffix}/${architectures_list_b[$i]::-1}
		mode=${mode_list[i]}
		mkdir -p ${save_suffix}/${model_suffix}
		model=${architectures_list_b[$i]}${act}${architectures_list_e[$i]}
		for drop_rate in ${drop_rates}
		do
			for lr in ${learning_rates_list}
			do
				for run in $(seq 0 $run_max)
				do
					mkdir -p ${save_suffix}/${model_suffix}/${lr}-${drop_rate}/r${run}
					echo python3 ${python_script} --train_data ${training_set} --val_data ${validation_set} --test_data ${test_set} --batch_size ${batch_size} --val_batch_size ${batch_val_size} --input_dim 5 --output_dim 3 --dropout ${drop_rate} --model ${model} --learning_rate ${lr} --timestamp_idx 0 --output ${save_suffix}/${model_suffix}/${lr}-${drop_rate}/r${run} --tb_dir ${tb_suffix}/${model_suffix} --tb_log_name ${lr}-${drop_rate}-r${run} --reader_mode ${mode} --trajectory_length=${traj_length} --max_iterations ${iterations} --max_sequence_size ${used_points}>> ${to_file}
				done
			done
		done
	done
done

ds_suffix=${DS_2_suffix}
mkdir -p ${save_suffix}/${ds_suffix}
for (( i=0; i<${#architectures_list_b[*]}; i++ ))
do
	model_suffix=${ds_suffix}/${architectures_list_b[$i]::-1}
	mkdir -p ${save_suffix}/${model_suffix}
	mode=${mode_list[$i]}
	for drop_rate in ${drop_rates}
	do
		model=${architectures_list_b[$i]}${act}${architectures_list_e[$i]}
		for lr in ${learning_rates_list}
		do
			for run in $(seq 0 $run_max)
			do
				mkdir -p ${save_suffix}/${model_suffix}/${lr}-${drop_rate}/r${run}
				echo python3 ${python_script} --train_data ${training_set} --val_data ${validation_set} --test_data ${test_set} --batch_size ${batch_size} --val_batch_size ${batch_val_size} --input_dim 5 --output_dim 3 --dropout ${drop_rate} --model ${model} --learning_rate ${lr} --timestamp_idx 0 --output ${save_suffix}/${model_suffix}/${lr}-${drop_rate}/r${run} --tb_dir ${tb_suffix}/${model_suffix} --tb_log_name ${lr}-${drop_rate}-r${run} --reader_mode ${mode} --trajectory_length=${traj_length} --max_iterations ${iterations} --max_sequence_size ${used_points}>> ${to_file}
			done
		done
	done
done

for (( j=0; j<${#training_set_3[*]}; j++ ))
do
	training_set=${training_set_3[$j]}
	ds_suffix=${DS_3_suffix[$j]}
	mkdir -p ${save_suffix}/${ds_suffix}
	for (( i=0; i<${#architectures_list_b[*]}; i++ ))
	do
		model_suffix=${ds_suffix}/${architectures_list_b[$i]::-1}
		mkdir -p ${save_suffix}/${model_suffix}
		mode=${mode_list[$i]}
		for drop_rate in ${drop_rates}
		do
			model=${architectures_list_b[$i]}${act}${architectures_list_e[$i]}
			for lr in ${learning_rates_list}
			do
				for run in $(seq 0 $run_max)
				do
					mkdir -p ${save_suffix}/${model_suffix}/${lr}-${drop_rate}/r${run}
					echo python3 ${python_script} --train_data ${training_set} --val_ratio 0.1 --test_ratio 0.1 --batch_size ${batch_size} --val_batch_size 1000 --input_dim 8 --output_dim 4 --dropout ${drop_rate} --model ${model} --learning_rate ${lr} --output ${save_suffix}/${model_suffix}/${lr}-${drop_rate}/r${run} --tb_dir ${tb_suffix}/${model_suffix} --tb_log_name ${lr}-${drop_rate}-r${run} --reader_mode ${mode} --trajectory_length=${traj_length} --max_iterations ${iterations} --max_sequence_size ${used_points}>> ${to_file}
				done
			done
		done
	done
done
