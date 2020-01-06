# GRID SEARCH PARAMETERS
alpha_max=9
alpha_min=1
beta_max=9
beta_min=1

# VALIDATION PARAMETERS
run_max=4

# TRAINING PARAMETERS
iterations=30000
expererience_replay=2000
used_points=12
pred_points=1
batch_size=64
batch_test_size=10000
batch_val_size=10000
drop_rate=0.1
model= MLP_LRELU_d32_d32
learning_rate=0.005
traj_length=20

# PATH
root="/cs-share/dream/systemid_v2"
training_set="$root/data/simu/train-full-random"
validation_set="$root/data/simu/val"
test_set="$root/data/simu/test"
python_script="$root/networks/run_training.py"

# OUTPUT
output_root="/cs-share/dream/RSS-2020-Priorization"
to_file="$output_root/gridsearch_per_full_random.txt"
save_suffix="$output_root/results/GS/PER_full_random"
tb_suffix="$output_root/tensorboard/GS/PER_full_random"

# INIT
mkdir -p $output_suffix

for alpha in $(seq $alpha_min $alpha_max)
do
	for beta in $(seq $beta_min $beta_max)
	do
		for run in $(seq 0 $run_max)
		do
			mkdir -p ${save_suffix}/0{$alpha}_0${beta}/r{$run}
			echo python3 ${python_script} --train_data ${training_set} --val_set ${validation_set} --test_set ${test_set} --batch_size ${batch_size} --val_batch_size ${batch_val_size} -- test_batch_size ${batch_test_size} --input_dim 5 --output_dim 3 --dropout ${drop_rate} --model ${model} --learning_rate ${learning_rate} --timestamp_idx 0 --output ${save_suffix}/0{$alpha}_0${beta}/r{$run} --tb_dir ${tb_suffix} --tb_log_name 0{$alpha}_0${beta}-r{$run} --reader_mode classic --priorization PER --trajectory_lenght=${traj_length} --alpha ${alpha} --beta ${beta} >> ${to_file}
		done
	done
done

