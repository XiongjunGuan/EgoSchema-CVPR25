import argparse
import json
import os
import os.path as osp
import re
import sys
import time
from collections import Counter

import json_repair
import numpy as np


# 去冗余加权投票
def decorrelated_weighted_vote(answers, accuracies, similarities, n_choices):
    n = len(answers)
    effective_weights = []
    for i in range(n):
        redundancy = sum(similarities[i][j] for j in range(n) if j != i)
        w = accuracies[i] / (1 + redundancy)
        effective_weights.append(w)
    scores = np.zeros(n_choices)
    for ans, w in zip(answers, effective_weights):
        scores[int(ans)] += w
    best_choice = np.argmax(scores)
    return best_choice, scores, effective_weights


# 简单加权投票
def weighted_majority_vote(answers, accuracies, n_choices):
    scores = np.zeros(n_choices)
    for ans, acc in zip(answers, accuracies):
        scores[int(ans)] += acc
    best_choice = np.argmax(scores)
    return best_choice, scores


if __name__ == "__main__":

    expert_path_lst = [
        "./egoschema-answers/submission_1.json",
        "./egoschema-answers/submission_2.json",
        "./egoschema-answers/submission_3.json",
        "./egoschema-answers/submission_4.json",
        "./egoschema-answers/submission_5.json",
        "./egoschema-answers/submission_6.json",
        "./egoschema-answers/submission_7.json",
    ]
    expert_acc = [0.759, 0.737, 0.752, 0.740, 0.730, 0.744, 0.737]

    # 结果字典
    experts_dict = {}
    cnt = 0
    for expert_path in expert_path_lst:
        with open(expert_path, "r") as file:
            expert = json.load(file)
        expert = dict(expert)
        experts_dict[cnt] = expert
        cnt += 1

    # 统计专家信息，检查专家之间的相似度
    all_res = []
    for k in experts_dict[0].keys():
        arr = []
        for expert_idx in experts_dict.keys():
            arr.append(experts_dict[expert_idx][k])
        all_res.append(arr)

    all_res = np.array(all_res)
    ex_keys = experts_dict.keys()
    expert_num = len(ex_keys)
    sim_arr = np.ones((expert_num, expert_num))
    for i in range(expert_num):
        for j in range(expert_num):
            sim = all_res[:, i] == all_res[:, j]
            sim = np.mean(sim * 1.0)
            sim_arr[i, j] = sim

    # 考虑相似度矩阵的多数投票
    records = []
    final_res = {}

    cnt = 0
    for k in experts_dict[0].keys():
        answer = []
        for expert_idx in range(expert_num):
            answer.append(experts_dict[expert_idx][k])

        choice, _, _ = decorrelated_weighted_vote(
            answer, expert_acc, sim_arr, n_choices=5
        )

        # weighted_majority_vote 或者 decorrelated_weighted_vote
        tmp, _ = weighted_majority_vote(answer, expert_acc, n_choices=5)

        if tmp != choice:
            cnt += 1

        final_res[k] = str(choice)

    print(cnt)
    # 将字典写入 JSON 文件
    with open("./rethinking/vote_sim.json", "w") as json_file:
        json.dump(final_res, json_file, indent=4)
