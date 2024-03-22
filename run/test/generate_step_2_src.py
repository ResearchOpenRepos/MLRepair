import os
import json
import shutil
if __name__ == "__main__":
	for level in ["coarse", "medium", "fine"]:
		test_src_path = "../../dataset/{}/patch_generation/src-test.jsonl".format(level)
		test_tgt_path = "../../dataset/{}/patch_generation/tgt-test.jsonl".format(level)
		if level == "coarse":
			beam = 1
		elif level == "medium":
			beam = 4
		elif level == "fine":
			beam = 10
		template_pred_path = "./{}/predict_tp/test_checkpoint-best-bleu_{}.output".format(level, beam)
		out_dir = "./{}/step_2_data/".format(level)
		os.makedirs(out_dir, exist_ok=True)
		with open(test_src_path, "r") as file:
			ori_srcs = file.readlines()
		with open(test_tgt_path, "r") as file:
			ori_tgts = file.readlines()
		with open(template_pred_path, "r") as file:
			pred_tps = file.readlines()
			pred_tps = [line.strip() for line in pred_tps]
		new_srcs = []
		new_tgts = []
		for i in range(len(ori_srcs)):
			src = ori_srcs[i].strip()
			src = json.loads(src)["INPUT"]
			tgt = ori_tgts[i].strip()
			for j in range(i*beam, (i+1)*beam):
				pred_tp = pred_tps[j]
				new_src_tokens = ["#"] + [pred_tp] + src.split(" ")[2:]
				new_src_string = " ".join(new_src_tokens)
				new_src = {"INPUT": repr(new_src_string)[1:-1].replace("\\n", "\n")}
				new_srcs.append(new_src)
				new_tgts.append(tgt)
		with open(os.path.join(out_dir, "src-test.jsonl"), "w") as file:
			for line in new_srcs:
				json.dump(line, file)
				file.write("\n")
		with open(os.path.join(out_dir, "tgt-test.jsonl"), "w") as file:
			file.write("\n".join(new_tgts))
	
