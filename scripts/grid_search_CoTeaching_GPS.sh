# GRID SEARCH PARAMETERS
tau_list="0.0125 0.025 0.0375 0.05 0.0625 0.075 0.0875 0.1 0.1125 0.125 0.1375 0.15 0.1625 0.175 0.2 0.25"
k_iter_list="10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000"

# VALIDATION PARAMETERS
run_max=4

# TRAINING PARAMETERS
iterations=150000
used_points=12
pred_points=1
batch_size=128
batch_val_size=10000
drop_rate=0.1
model="MLP_LRELU_d32_d32"
learning_rate=0.0025
traj_length=20

# PATH
root="/cs-share/dream/systemid_v2"
data_root="/cs-share/dream/kf_ws/src/lake_wanderer/records"
training_set="$data_root/GPS_PROCESSED_NH"
validation_set="$data_root/FIX_VAL"
test_set="$data_root/FIX_TEST"
python_script="$root/networks/run_training.py"

# OUTPUT
output_root="/cs-share/dream/RSS-2020-Priorization"
to_file="$output_root/gridsearch_coteaching_GPS_v3.txt"
save_suffix="$output_root/results/GS/CoTeachingGPS_v3"
tb_suffix="$output_root/tensorboard/GS/CoTeachingGPS_v3"

# INIT
mkdir -p ${output_root}

for tau in ${tau_list}
do
	for k_iter in ${k_iter_list}
	do
		for run in $(seq 0 $run_max)
		do
			mkdir -p ${save_suffix}/${tau}-${k_iter}/r${run}
			echo python3 ${python_script} --train_data ${training_set} --val_data ${validation_set} --test_data ${test_set} --batch_size ${batch_size} --val_batch_size ${batch_val_size} --input_dim 5 --output_dim 3 --dropout ${drop_rate} --model ${model} --learning_rate ${learning_rate} --timestamp_idx 0 --output ${save_suffix}/${tau}-${k_iter}/r${run} --tb_dir ${tb_suffix} --tb_log_name ${tau}-${k_iter}-r${run} --reader_mode co_teaching --trajectory_length=${traj_length} --k_iter ${k_iter} --tau ${tau} --max_iterations ${iterations} >> ${to_file}
		done
	done
done

