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
from dotenv import load_dotenv
from pathlib import Path

# 加载 .env 文件
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

def get_env_var(key, default=None):
    """获取环境变量，如果不存在则返回默认值"""
    return os.getenv(key, default)

def get_env_bool(key, default=False):
    """获取布尔类型的环境变量"""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

def get_env_int(key, default=0):
    """获取整数类型的环境变量"""
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default

# 项目配置
PROJECT_ID = get_env_var("PROJECT_ID")
SPEECH_PROJECT_ID = get_env_var("SPEECH_PROJECT_ID")
IMAGE_PROJECT_ID = get_env_var("IMAGE_PROJECT_ID")
VIDEO_PROJECT_ID = get_env_var("VIDEO_PROJECT_ID")
GCS_PROJECT_ID = get_env_var("GCS_PROJECT_ID")
LOCATION = get_env_var("LOCATION")

# 服务账户配置
SA_KEY_FILE_PATH = get_env_var("SA_KEY_FILE_PATH")

# API Key 配置
API_KEY = get_env_var("API_KEY")

# 认证配置
GEMINI_AUTH_TYPE = get_env_var("GEMINI_AUTH_TYPE", "ADC")
SPEECH_AUTH_TYPE = get_env_var("SPEECH_AUTH_TYPE", "ADC")
IMAGE_AUTH_TYPE = get_env_var("IMAGE_AUTH_TYPE", "ADC")
VIDEO_AUTH_TYPE = get_env_var("VIDEO_AUTH_TYPE", "ADC")
GCS_AUTH_TYPE = get_env_var("GCS_AUTH_TYPE", "ADC")

# 超时配置
GEMINI_TIMEOUT = get_env_int("GEMINI_CLIENT_TIMEOUT")

# MiniMax 配置
MINIMAX_GROUP_ID = get_env_var("MINIMAX_GROUP_ID")
MINIMAX_API_KEY = get_env_var("MINIMAX_API_KEY")

# Movie clip 参数
MC_PARTITION_SECONDS = get_env_int("MC_PARTITION_SECONDS", 600)
MC_BUCKET_NAME = get_env_var("MC_BUCKET_NAME")
MC_OBJECT_PREFIX = get_env_var("MC_OBJECT_PREFIX")
MC_NEED_GCS = get_env_bool("MC_NEED_GCS",True)

# 其他配置
FFMPEG_EXECUTABLE = get_env_var("FFMPEG_EXECUTABLE", "ffmpeg")
FFPROBE_EXECUTABLE = get_env_var("FFPROBE_EXECUTABLE", "ffprobe")
PROXY = get_env_var("PROXY")
