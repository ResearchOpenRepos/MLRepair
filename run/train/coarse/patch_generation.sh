# configuration

lr=5e-5
batch_size=16
beam_size=1
source_length=512
target_length=256
output_dir=saved_models_pg
input_dir=../../../dataset/coarse/patch_generation
train_file=$input_dir/src-train.jsonl,$input_dir/tgt-train.jsonl
dev_file=$input_dir/src-val.jsonl,$input_dir/tgt-val.jsonl
log_file=train.log
epochs=30
pretrained_model=../../../unixcoder-base

mkdir -p $output_dir

python ../../run.py --do_train --do_eval --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs 2>&1| tee $output_dir/$log_file

