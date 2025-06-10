import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import json_repair
import numpy as np
from google import genai


def process_video_qa(video_path, api_key, question, options):

    # 构建选项文本
    options_text = ""
    for i, option in enumerate(options):
        options_text += f"option {i}: {option}\n"

    # 构建提示文本

    prompt_template = """
I am working on a video question answering task. 
Given a video, a question, and several answer options, I need to identify the option that best aligns with the question and the video content. 
To better understand the video and make a more informed decision, I would like to generate prompt words for the content that needs to be closely observed based on the question and options.
These prompts will help Gemini understand the key aspects to pay attention to and the differences between the options. 
I will include these texts in the prompts for Gemini.

[QUESTION]
{question}

[OPTION]
{options_text}

[RULES]
Based on the above text, please give the content that should be focused on in the video.

    """

    prompt = prompt_template.format(question=question, options_text=options_text)
    # print(prompt)

    # 初始化上传器并处理视频
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    response = response.text

    result = {"PROMPT": response}

    return result


def process_item(
    record_video,
    item,
    api_key,
    video_dir,
    output_dir,
    sleep_time=0.1,
):
    time.sleep(sleep_time)

    video_path = f"{video_dir}/{record_video}.mp4"
    if not os.path.exists(video_path):
        print(f"视频不存在: {video_path}")
        return

    output_file = f"{output_dir}/{record_video}.json"
    # 跳过已处理的文件
    if os.path.exists(output_file):
        print(f"跳过已处理: {record_video}")
        return

    # 提取问题和选项
    uid = item["q_uid"]
    question = item["question"]
    options = [
        item["option 0"],
        item["option 1"],
        item["option 2"],
        item["option 3"],
        item["option 4"],
    ]

    try:
        # 处理视频问答
        result = process_video_qa(video_path, api_key, question, options)
        # 保存结果到文件
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"已处理: {uid}")
    except Exception as e:
        print(f"处理出错 [{api_key}] - {uid}: {e}")

    return


def batch_process_questions(
    questions_file,
    video_dir,
    api_keys,
    output_dir,
    max_process,
    wait_time,
):

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载问题数据
    with open(questions_file, "r") as f:
        questions = json.load(f)

    # 分配API密钥
    num_keys = len(api_keys)

    # 生成输入参数表
    args_list = []
    cnt = 0
    cnt_processed = 0
    for item in questions:
        api_key = api_keys[cnt % num_keys]
        record_video = item["q_uid"]

        if os.path.exists(f"{output_dir}/{record_video}.json"):
            cnt_processed += 1
            continue

        args_list.append((record_video, item, api_key, video_dir, output_dir))
        cnt += 1
    num_all_samples = len(args_list)

    max_process = min(max_process, num_keys)

    print("\n\n========== {} samples has be processed. ".format(cnt_processed))
    print("========== {} samples need to be processed. \n".format(num_all_samples))
    print(
        "========== {} keys in total, max_workers={}, each consuming an average of {} request. \n".format(
            len(api_keys),
            max_process,
            num_all_samples // len(api_keys),
        )
    )

    args_start_idx = 0
    while args_start_idx < len(args_list):
        args_end_idx = args_start_idx + max_process
        args_end_idx = min(args_end_idx, len(args_list))
        args_subset = args_list[args_start_idx:args_end_idx]

        t1 = time.time()

        # 创建进程池
        max_process = min(len(args_subset), max_process)

        # 使用ThreadPoolExecutor进行多线程处理
        with ThreadPoolExecutor(max_workers=num_keys) as executor:
            futures = []
            prompt_idx = 1
            for args in args_subset:
                record_video, item, api_key, video_dir, output_dir = args
                sleep_time = prompt_idx - 1
                futures.append(
                    executor.submit(
                        process_item,
                        record_video,
                        item,
                        api_key,
                        video_dir,
                        output_dir,
                        sleep_time,
                    )
                )

                prompt_idx += 1
                if prompt_idx > max_process:
                    prompt_idx = 1

            for future in as_completed(futures):
                future.result()  # 等待所有任务完成

        t2 = time.time()
        t = t2 - t1
        if t < wait_time:
            time.sleep(wait_time - t)
        time.sleep(0.5)

        t3 = time.time()

        args_start_idx = args_end_idx

        print(
            "======== batch done: {} samples, {:.1f} s".format(
                len(args_subset),
                t3 - t1,
            )
        )
        num_remain_samples = num_all_samples + 1 - args_start_idx
        print(
            "======== {} items not yet processed, need about {:.1f} hour. \n".format(
                num_remain_samples,
                np.ceil(num_remain_samples / max_process) * (t3 - t1) / 3600,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用Gemini处理视频问答")
    parser.add_argument(
        "--questions_file",
        default="./questions/questions.json",
        help="问题JSON文件路径",
    )
    parser.add_argument(
        "--videos_dir",
        default="../competition/EgoSchema/videos",
        help="视频目录",
    )

    parser.add_argument(
        "--output_dir",
        default="./results/2stage/gemini_attention",
        help="结果保存的文件路径",
    )
    parser.add_argument(
        "--max_process",
        default=2,
        help="最大线程数, 逐个batch运行",
    )
    parser.add_argument(
        "--wait_time",
        default=25,
        help="每个batch的最小时间间隔",
    )

    args = parser.parse_args()

    api_keys = []

    for i in range(10):
        batch_process_questions(
            args.questions_file,
            args.videos_dir,
            api_keys,
            args.output_dir,
            int(args.max_process),
            int(args.wait_time),
        )
        time.sleep(30)
