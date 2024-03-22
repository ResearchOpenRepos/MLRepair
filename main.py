import os

if __name__ == "__main__":
	# 1. Model fine-tuning for tempate prediction sub-task.
	os.system("cd ./run/train/coarse/ && .\template_prediction.sh")
	os.system("cd ./run/train/medium/ && .\template_prediction.sh")
	os.system("cd ./run/train/fine/ && .\template_prediction.sh")
	
	# 2.  Model fine-tuning for patch generation sub-task.
	os.system("cd ./run/train/coarse/ && .\patch_generation.sh")
	os.system("cd ./run/train/medium/ && .\patch_generation.sh")
	os.system("cd ./run/train/fine/ && .\patch_generation.sh")
	os.system("cd ./run/train/no_template/ && .\patch_generation.sh")

	# 3. Get template prediction results via fine-tuned LLMs.
	os.system("cd ./run/test/coarse/ && .\template_prediction.sh")
	os.system("cd ./run/test/medium/ && .\template_prediction.sh")
	os.system("cd ./run/test/fine/ && .\template_prediction.sh")
	
	# 4. Assemble the predicted results into the src file for the next sub-task.
	os.system("cd ./run/test/ && python generate_step_2_src.py")

	# 5. Get patch generation results via fine-tuned LLMs.
	os.system("cd ./run/test/coarse/ && .\patch_generation.sh")
	os.system("cd ./run/test/medium/ && .\patch_generation.sh")
	os.system("cd ./run/test/fine/ && .\patch_generation.sh")
	os.system("cd ./run/test/no_template/ && .\patch_generation.sh")

	# 6. Patch combination to fuse multi-granularity templates.
	os.system("cd ./run/test/ && python combine_result.py")
	
	# 7. Generate final repair results.
	os.system("cd ./run/test/ && python generate_result.py")
