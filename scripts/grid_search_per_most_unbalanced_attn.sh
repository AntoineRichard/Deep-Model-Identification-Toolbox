# GRID SEARCH PARAMETERS
alpha_max=9
alpha_min=1
beta_max=9
beta_min=1

# VALIDATION PARAMETERS
run_max=4

# TRAINING PARAMETERS
iterations=30000
per_refresh=1000
used_points=6
pred_points=1
batch_size=64
batch_test_size=10000
batch_val_size=10000
drop_rate=0.1
model="ATTNMPMH_LRELU_a1.75_md32_d32"
learning_rate=0.001
traj_length=20

# PATH
root="/cs-share/dream/systemid_v2"
training_set="$root/data/simu/train-most-unbalanced"
validation_set="$root/data/simu/val"
test_set="$root/data/simu/test"
python_script="$root/networks/run_training.py"

# OUTPUT
output_root="/cs-share/dream/RSS-2020-Priorization"
to_file="$output_root/gridsearch_per_most_unbalanced_attn.txt"
save_suffix="$output_root/results/GS/PER_most_unbalanced_attn"
tb_suffix="$output_root/tensorboard/GS/PER_most_unbalanced_attn"

# INIT
mkdir -p $output_root

for alpha in $(seq $alpha_min $alpha_max)
do
	for beta in $(seq $beta_min $beta_max)
	do
		for run in $(seq 0 $run_max)
		do
			mkdir -p ${save_suffix}/0${alpha}_0${beta}/r${run}
			echo python3 ${python_script} --train_data ${training_set} --val_data ${validation_set} --test_data ${test_set} --batch_size ${batch_size} --val_batch_size ${batch_val_size} --test_batch_size ${batch_test_size} --input_dim 5 --output_dim 3 --dropout ${drop_rate} --model ${model} --learning_rate ${learning_rate} --timestamp_idx 0 --output ${save_suffix}/0${alpha}_0${beta}/r${run} --tb_dir ${tb_suffix} --tb_log_name 0${alpha}_0${beta}-r${run} --reader_mode seq2seq --priorization PER --trajectory_length ${traj_length} --per_refresh_rate ${per_refresh} --update_batchsize 50000 --alpha 0.${alpha} --beta 0.${beta} --max_iterations ${iterations} --max_sequence_size ${used_points} >> ${to_file}
		done
	done
done

