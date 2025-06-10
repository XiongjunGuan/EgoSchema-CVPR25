import time

from google import genai


class GoogleGenAIUploader:
    def __init__(self, api_key, prompt):
        self.client = genai.Client(api_key=api_key)
        self.prompt = prompt

    def upload_and_process(self, file_path):
        """同步上传文件"""
        print(f"Uploading file: {file_path}...")
        video_file = self.client.files.upload(file=file_path)
        print(f"Completed upload: {video_file.uri}")

        t_start = time.time()
        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            video_file = self.client.files.get(name=video_file.name)

            if time.time() - t_start > 60:
                raise ValueError("Time out, Waiting time exceeds 60s")

        if video_file.state.name == "FAILED":
            raise ValueError("File processing failed")

        return video_file

    def generate_summary(self, video_file):
        """同步生成摘要"""
        try:
            response = self.client.models.generate_content(
                model="models/gemini-2.0-flash",
                contents=[video_file, self.prompt],
            )
        except Exception as e:
            # 删除文件
            self.client.files.delete(name=video_file.name)
            print(f"{video_file.name} has been deleted")
            raise ValueError(e)

        self.client.files.delete(name=video_file.name)
        print(f"{video_file.name} has been deleted")

        return response.text

    def process_files(self, file_paths):
        """处理文件并返回结果"""
        results = []
        for file_path in file_paths:
            try:
                video_file = self.upload_and_process(file_path)
                summary = self.generate_summary(video_file)
                results.append((file_path, summary))
            except Exception as e:
                raise ValueError(e)

        return results
