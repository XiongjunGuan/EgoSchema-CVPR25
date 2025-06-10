import json
import os

result_dir = "./results/1stage/gemini_v2"
output_path = "./egoschema-answers/submission_1.json"

# 读取所有问题
with open("./questions/questions.json", "r") as f:
    questions = json.load(f)

# 初始化结果字典
results = {}

# 遍历所有问题
for question in questions:
    file_id = question["q_uid"]

    # 检查路径
    first_path = os.path.join(
        result_dir,
        f"{file_id}.json",
    )
    if os.path.exists(first_path):
        with open(first_path, "r") as f:
            data = json.load(f)
            answer = data.get("ANSWER")
            if answer is not None:
                results[file_id] = str(answer)
                # print(f"Found in first path: {file_id}")
                continue

    print(f"Warning: No answer found for question {file_id}")

# 将结果写入submission.json文件
with open(output_path, "w") as f:
    json.dump(results, f)

print(f"Total questions: {len(questions)}")
print(f"Total answers found: {len(results)}")
