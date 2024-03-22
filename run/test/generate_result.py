import os
import json

with open("./final_result/combine_patch.txt", "r") as file:
	preds = file.readlines()
	preds = [line.strip() for line in preds]
with open("./final_result/combine_template.txt", "r") as file:
	templates = file.readlines()
	templates = [line.strip() for line in templates]
with open("../../dataset/fine/patch_generation/tgt-test.jsonl", "r") as file:
	tgts = file.readlines()
	tgts = [json.loads(line)["INPUT"] for line in tgts]
beam = 100
count = 0
correct_templates = []
correct_patches = []
for i in range(len(tgts)):
	pred = preds[i*beam:(i+1)*beam]
	template = templates[i*beam:(i+1)*beam]
	for j, factor in enumerate(pred):
		if factor == tgts[i]:
			count += 1
			correct_patches.append("{} {}".format(i+1, j+1))
			correct_templates.append(template[j])
			break
with open("patch_correct.txt", "w") as file:
	file.write("\n".join(correct_patches))
with open("template_correct.txt", "w") as file:
	file.write("\n".join(correct_templates))
print(count)
