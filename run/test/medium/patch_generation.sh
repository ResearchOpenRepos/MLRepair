# configuration

lr=5e-5
batch_size=16
beam_size=25
output_size=25
source_length=512
target_length=256
saved_model_dir=../../train/medium/saved_models_pg
output_dir=predict_pg
input_dir=step_2_data
dev_file=None
test_file=$input_dir/src-test.jsonl,$input_dir/tgt-test.jsonl
checkpoint_type=best-bleu # best-ppl/bleu last
load_model_path=$saved_model_dir/checkpoint-$checkpoint_type/pytorch_model.bin
log_file=test_${checkpoint_type}_${output_size}.log
pretrained_model=../../../unixcoder-base


python ../../../run.py --do_test --model_name_or_path $pretrained_model --load_model_path $load_model_path --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --output_size $output_size --eval_batch_size $batch_size 2>&1| tee $output_dir/$log_file
