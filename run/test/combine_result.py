import os
import ast

def check_syntax(s):
	s = s.strip()
	s = s.split("\\n")
	s = [line.strip() for line in s]
	new_s = []
	for line in s:
		if line.endswith(":"):
			new_s += [line, "    pass"]
		else:
			new_s += [line]
	if len(new_s) == 1 and new_s[0].startswith("@"):
		new_s = [new_s[0], "def func():", "    pass"]
	elif new_s[0].startswith("elif") or new_s[0].startswith("else"):
		new_s = ["if True:", "    pass"] + new_s
	elif new_s[0].startswith("except "):
		new_s = ["try:", "    pass"] + new_s
	new_s = "\n".join(new_s)
	try:
		ast.parse(new_s)
		return True
	except:
		return False

fine_path = "./fine/predict_pg/patch_100.output"
medium_path = "./medium/predict_pg/patch_100.output"
coarse_path = "./coarse/predict_pg/patch_100.output"
no_template_path = "./no_template/predict_pg/patch_100.output"

fine_template_path = "./fine/predict_tp/test_checkpoint-best-bleu_10.output"
medium_template_path = "./medium/predict_tp/test_checkpoint-best-bleu_4.output"
coarse_template_path = "./coarse/predict_tp/test_checkpoint-best-bleu_1.output"

with open(fine_path, "r") as file:
	content_1 = file.readlines()
	content_1 = [line.strip() for line in content_1]
with open(medium_path, "r") as file:
	content_2 = file.readlines()
	content_2 = [line.strip() for line in content_2]
with open(coarse_path, "r") as file:
    content_3 = file.readlines()
    content_3 = [line.strip() for line in content_3]
with open(no_template_path, "r") as file:
    content_4 = file.readlines()
    content_4 = [line.strip() for line in content_4]

with open(fine_template_path, "r") as file:
	template_1 = file.readlines()
	template_1 = [line.strip() for line in template_1]
	new_template_1 = []
	for factor in template_1:
		new_template_1 += ["1. " + factor] * 10
	template_1 = new_template_1
with open(medium_template_path, "r") as file:
	template_2 = file.readlines()
	template_2 = [line.strip() for line in template_2]
	new_template_2 = []
	for factor in template_2:
		new_template_2 += ["2. " + factor] * 25
	template_2 = new_template_2
with open(coarse_template_path, "r") as file:
	template_3 = file.readlines()
	template_3 = [line.strip() for line in template_3]
	new_template_3 = []
	for factor in template_3:
		new_template_3 += ["3. " + factor] * 100
	template_3 = new_template_3
template_4 = ["4. no_template"] * len(content_4)

templates = [list(item) for item in zip(template_1, template_2, template_3, template_4)]
beam = 100
new_preds = []
new_templates = []
for i in range(len(content_1) // beam):
	t1 = content_1[i*beam:(i+1)*beam]
	t2 = content_2[i*beam:(i+1)*beam]
	t3 = content_3[i*beam:(i+1)*beam]
	t4 = content_4[i*beam:(i+1)*beam]
	combine = []
	combine_template = []
	for index, curr in enumerate(zip(t1,t2,t3,t4)):
		for template_index, factor in enumerate(curr):
			if len(combine) == beam:
				break
			if factor not in combine:
				if(check_syntax(factor)):
					combine.append(factor)
					combine_template.append(templates[i*beam+index][template_index])
				else:
					#print(factor)
					pass
	if len(combine) < beam:
		combine += [""]*(beam-len(combine))
		combine_template += ["empty"]*(beam-len(combine_template))
	new_preds += combine
	new_templates += combine_template

os.makedirs("./final_result/", exist_ok=True)
with open("./final_result/combine_patch.txt", "w") as file:
	file.write("\n".join(new_preds))
with open("./final_result/combine_template.txt", "w") as file:
	file.write("\n".join(new_templates))
