# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import List, Dict
import concurrent.futures
from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account
from ailib.env_config import GCS_PROJECT_ID, SA_KEY_FILE_PATH, GCS_AUTH_TYPE


def _get_gcs_credentials():
  """
  根据 GCS_AUTH_TYPE 获取 GCS 凭据
  """
  credentials = None
  if GCS_AUTH_TYPE == "SA":
    if SA_KEY_FILE_PATH and os.path.exists(SA_KEY_FILE_PATH):
      credentials = service_account.Credentials.from_service_account_file(
          SA_KEY_FILE_PATH,
          scopes=['https://www.googleapis.com/auth/cloud-platform']
      )
    else:
      print(f"警告: GCS SA 认证模式但服务账户文件不存在: {SA_KEY_FILE_PATH}")
      print("回退到默认认证模式 (ADC)")
  else:
    print(f"GCS 使用默认认证模式 (ADC)")

  return credentials


def init_client():
  """
  初始化 GCS 客户端
  根据 GCS_PROJECT_ID 和 GCS_AUTH_TYPE 配置
  """
  project_id = GCS_PROJECT_ID
  credentials = _get_gcs_credentials()

  storage_client = storage.Client(
      project=project_id,
      credentials=credentials
  )
  return storage_client


def download_gcs_object(gcs_uri: str,
    local_destination_path: str = None) -> bool:
  """
  从 Google Cloud Storage 下载一个对象到本地文件系统。

  参数:
      gcs_uri (str): GCS 对象的完整 URI，格式为 "gs://bucket-name/path/to/object".
      local_destination_path (str, optional): 本地保存路径，包括文件名。
                                              如果为 None，将使用 GCS 对象的名称，并保存在当前工作目录。

  返回:
      bool: 如果下载成功则返回 True，否则返回 False。
  """
  if not gcs_uri.startswith("gs://"):
    print(f"错误：无效的 GCS URI '{gcs_uri}'。它必须以 'gs://' 开头。")
    return False

  try:
    # 解析 GCS URI
    path_parts = gcs_uri[5:].split("/", 1)
    if len(path_parts) < 2 or not path_parts[0] or not path_parts[1]:
      print(
        f"错误：无效的 GCS URI 格式 '{gcs_uri}'。期望格式: gs://bucket-name/object-path")
      return False

    bucket_name = path_parts[0]
    blob_name = path_parts[1]

    # 如果 blob_name 以 '/' 结尾，说明它可能是一个“文件夹”占位符，或者用户错误地指定了
    # download_to_filename 不能下载这样的对象，它需要一个具体的文件名。
    if blob_name.endswith('/'):
      print(
        f"错误：对象路径 '{blob_name}' 看起来像一个目录。请指定一个具体的文件对象。")
      return False

    # 初始化 GCS 客户端
    storage_client = init_client()

    # 获取 bucket 和 blob
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # 确定本地目标路径
    if local_destination_path is None:
      local_destination_path = os.path.basename(blob_name)
      if not local_destination_path:  # 防止 blob_name 是类似 "some/path/" 后又被错误处理到这里
        print(f"错误：无法从 GCS 路径 '{blob_name}' 推断出本地文件名。")
        return False
    else:
      # 确保目标目录存在
      local_dir = os.path.dirname(local_destination_path)
      if local_dir and not os.path.exists(local_dir):
        os.makedirs(local_dir)
        print(f"创建目录: {local_dir}")

    print(
      f"正在下载 gs://{bucket_name}/{blob_name} 到 {local_destination_path}...")

    blob.download_to_filename(local_destination_path)

    print(f"文件成功下载到: {local_destination_path}")
    return True

  except NotFound:
    print(
      f"错误：在 GCS 上未找到对象 gs://{bucket_name}/{blob_name} 或存储桶不存在。")
    return False
  except Exception as e:
    print(f"下载过程中发生错误: {e}")
    return False


def upload_gcs_object(bucket_name, destination_blob_name,
    source_file_name) -> bool:
  try:
    # 初始化 GCS 客户端
    storage_client = init_client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name,
                              if_generation_match=generation_match_precondition)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )
    return True
  except Exception as e:
    print(f"上传过程中发生错误: {e}")
    return False


def blob_metadata(bucket_name, blob_name):
  # 初始化 GCS 客户端
  storage_client = init_client()
  bucket = storage_client.bucket(bucket_name)

  # Retrieve a blob, and its metadata, from Google Cloud Storage.
  # Note that `get_blob` differs from `Bucket.blob`, which does not
  # make an HTTP request.
  blob = bucket.get_blob(blob_name)
  return blob


def _upload_task(local_file: str, bucket: storage.Bucket, gcs_path: str):
  """
  单个文件的上传任务，用于在线程池中执行。

  :return: 一个元组 (local_file, status_message)
  """
  try:
    file_name = Path(local_file).name
    # 拼接在 GCS 中的完整对象路径
    # 例如: 'videos/my_project/filename_part_1.mp4'
    object_name = f"{gcs_path}/{file_name}" if gcs_path else file_name
    object_gcs_name = f"gs://{bucket.name}/{object_name}"
    blob = bucket.blob(object_name)
    # 1. 判断文件是否已存在
    if blob.exists():
      status = "Skipped: Already exists"
      print(f"⏭️ {file_name}: {status}")
      return local_file, object_gcs_name, status

    # 2. 如果不存在，则上传
    print(f"🔼 {file_name}: Uploading...")
    blob.upload_from_filename(local_file, timeout=6000)
    status = "Uploaded"
    print(f"✅ {file_name}: {status}")
    return local_file, object_gcs_name, status

  except Exception as e:
    status = f"Failed: {e}"
    print(f"❌ {file_name}: {status}")
    return local_file, "", status


def upload_parts_to_gcs_parallel(
    local_files: List[str],
    bucket_name: str,
    object_path: str,
    max_workers: int = 8
) -> Dict[str, List[str]]:
  """
  并行地将一系列本地文件上传到 Google Cloud Storage。

  :param local_files: 要上传的本地文件路径列表。
  :param bucket_name: GCS 存储桶的名称。
  :param object_path: 文件在存储桶中存放的路径（"文件夹"）。
  :param max_workers: 并行上传的最大线程数。
  :return: 一个字典，键是本地文件路径，值是上传状态。
  """

  storage_client = init_client()
  bucket = storage_client.bucket(bucket_name)
  upload_results = {}

  # 使用线程池执行并行上传
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_workers) as executor:
    # 提交所有上传任务
    future_to_file = {
        executor.submit(_upload_task, local_file, bucket,
                        object_path): local_file
        for local_file in local_files
    }

    print(
      f"\n--- 开始并行上传到 GCS Bucket: gs://{bucket_name}/{object_path} ---")

    # 等待任务完成并收集结果
    for future in concurrent.futures.as_completed(future_to_file):
      try:
        file_path, object_name, status = future.result()
        upload_results[file_path] = [file_path, object_name, status]
      except Exception as e:
        # 一般 _upload_task 内部会捕获异常，这里作为最后的保障
        file_path = future_to_file[future]
        upload_results[file_path] = [file_path, object_name,
                                     f"Failed in executor: {e}"]

  return upload_results

# bucket_name = "gemini-oolongz"
# blob_name = "movie_metadata.txt"
# blob = blob_metadata(bucket_name, blob_name)
# print(blob)
