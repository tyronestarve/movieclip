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


import streamlit as st
import re
import os
import hashlib
import tempfile
import asyncio
import time
import random
from typing import Dict, List, Optional, Tuple, Any

from ailib import genai_client, voice_keys, genai_speech, video_process, \
  cloud_gcs, config
from google.genai import types
from ailib.env_config import (
  PROJECT_ID, LOCATION, SPEECH_PROJECT_ID,
  MC_PARTITION_SECONDS, MC_BUCKET_NAME, MC_OBJECT_PREFIX, MC_NEED_GCS,
  FFMPEG_EXECUTABLE, FFPROBE_EXECUTABLE
)

# https://gitlab.com/google-cloud-ce/googlers/oolongz/gemini-analysis-frontend/-/blob/f5d0d3fac46d45c2fabdf9c986483d12181628f5/src/social/pages/10_MovieHighLight.py
# Tips
# 1 audio_timestamp 对幻觉很重要，但是生成srt字幕时如果开启这个参数需要注意PE 
# 2 建议ffmpeg 升级到最新版本
# 3 project_root/data 用于cache中间文件，避免每次需要重新处理

# Constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
SUFFIX = "_gemini_clone_process"

# Color themes for different sections
COLOR_PARAMS = "#1f4e79"  # 蓝色系 - 参数设定区域
COLOR_PARAMS_GRAD = "#3970ac"
COLOR_OPERATIONS = "#2d5016"  # 绿色系 - 操作按钮区域  
COLOR_OPERATIONS_GRAD = "#70ad47"
COLOR_RESULTS = "#c55a11"  # 橙色系 - 结果展示区域
COLOR_RESULTS_GRAD = "#e36c09"
COLOR_DEBUG = "#5b2c6f"  # 紫色系 - Debug区域
COLOR_DEBUG_GRAD = "#7030a0"
COLOR_TEXT = "#ffffff"  # 白色文字

# Global configuration
VOICES = voice_keys.voice_clone_keys
FFMPEG_PATH = FFMPEG_EXECUTABLE
FFPROBE_PATH = FFPROBE_EXECUTABLE
PARTITION_SECONDS = MC_PARTITION_SECONDS
BUCKET_NAME = MC_BUCKET_NAME
OBJECT_PREFIX = MC_OBJECT_PREFIX
MODEL_LIST = config.gemini_long_output_model_list

# Session state keys
SESSION_KEYS = [
    "video_first_person", "mid_video", "audio_file_path",
    "final_output_mixed_audio_video", "final", "file_ori_local_path",
    "file_prefix",
    f"movie_plots_{SUFFIX}", f"movie_highlight_results_{SUFFIX}",
    f"movie_highlight_segments_{SUFFIX}", f"movie_highligh_video_{SUFFIX}",
    f"movie_highligh_video_narration_{SUFFIX}", f"movie_narration_{SUFFIX}",
    f"split_videos{SUFFIX}",
    f"highlight_starring_{SUFFIX}",
    f"actors_info_{SUFFIX}", f"role_set_{SUFFIX}", f"narration_style_{SUFFIX}",
    f"narration_character_nickname_{SUFFIX}", f"narration_movie_bg_{SUFFIX}",
    "tagging_results", "selected_genres", "tagging_explanations"
]

# Ensure data directory exists
if not os.path.exists(DATA_PATH):
  os.makedirs(DATA_PATH)


def streamlit_init():
  """Initialize Streamlit configuration and session state."""
  print("init")
  st.set_page_config(layout="wide")
  st.cache_data.clear()

  # Initialize session state keys
  for key in SESSION_KEYS:
    if key not in st.session_state:
      st.session_state[key] = None

  if "tagging_gcs_video_uri" not in st.session_state:
    st.session_state.tagging_gcs_video_uri = "gs://tcl-tv-video/m3u8.dev-tcl-badcase.mp4"
  
  if "tagging_content_type" not in st.session_state:
    st.session_state.tagging_content_type = "Movie"

  # Initialize specific keys
  if f"highlight_duration_{SUFFIX}" not in st.session_state:
    st.session_state[f"highlight_duration_{SUFFIX}"] = "20秒"
  if "process_error" not in st.session_state:
    st.session_state.process_error = ""
  if f"video_gcs_link{SUFFIX}" not in st.session_state:
    st.session_state[
      f"video_gcs_link{SUFFIX}"] = "gs://tcl-tv-video/m3u8.dev-tcl-badcase.mp4"
  if f"upload_{SUFFIX}" not in st.session_state:
    print("init upload")
    st.session_state[f"upload_{SUFFIX}"] = None
  if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

  st.session_state.init = True


def read_b64_from_file(filepath: str) -> Optional[bytes]:
  """Read file as bytes."""
  try:
    with open(filepath, "rb") as file:
      return file.read()
  except FileNotFoundError:
    print(f"Error: File not found at {filepath}")
    return None
  except Exception as e:
    print(f"An error occurred: {e}")
    return None


def write_string_to_file(file_path: str, string_data: str,
    encoding: str = 'utf-8') -> bool:
  """Write string to file with specified encoding."""
  try:
    print(f"write_string_to_file{file_path}")
    with open(file_path, 'w', encoding=encoding) as file:
      file.write(re.sub(r'^[^1\d]+', '', string_data))
    return True
  except Exception as e:
    print(f"Error writing string to file: {e}")
    return False


def write_bytes_tofile(uploaded_file, save_path: str) -> bool:
  """Write uploaded file bytes to local path."""
  try:
    with open(save_path, "wb") as f:
      f.write(uploaded_file.getbuffer())
    return True
  except Exception as e:
    print(f"Error writing bytes to file: {e}")
    return False


def calculate_md5(data) -> Optional[str]:
  """Calculate MD5 hash of given data."""
  try:
    if isinstance(data, str):
      data = data.encode('utf-8')

    md5_hash = hashlib.md5()
    md5_hash.update(data)
    return md5_hash.hexdigest()
  except Exception as e:
    print(f"Error calculating MD5: {e}")
    return None


def get_generation_config(temperature: float = 0.01, max_tokens: int = 65535) -> \
Dict[str, Any]:
  """Get common generation configuration."""
  return {
      "temperature": temperature,
      "response_mime_type": "text/plain",
      "max_output_tokens": max_tokens,
      "audio_timestamp": True
  }


def check_response_validity(response) -> Tuple[bool, object]:
  """Check if response is valid and return error message if not."""

  if response[3] == "fail":
    return False, response[0]
  if response[0].candidates[0].finish_reason != types.FinishReason.STOP:
    return False, response[0]
  return True, ""


def set_error_and_return(error_msg: str) -> None:
  """Set error message in session state."""
  st.session_state.process_error = error_msg


def gene_plots_by_splitvideos(split_videos: Dict,
    actor_description: str) -> Dict:
  """Generate plots by split videos using async execution."""
  return asyncio.run(gene_spots(split_videos, actor_description))


async def gene_spots(split_videos: Dict, actor_description: str) -> Dict:
  """Generate spots asynchronously."""
  tasks = {}
  model = st.session_state[f"model_{SUFFIX}"]

  prompt = f"""
    你是一位资深的电影剪辑师和预告片制作人。你的任务是分析视频片段，并根据严格的标准挑选出适合制作精彩预告片的剪辑素材。

    # 核心判断标准 (请综合以下几点进行艺术判断):
    1.  **关键情节节点**:
        * **主线任务触发**: 主角遇到重大挑战、与核心反派首次发生关键冲突等。
        * **高潮对决/冲突峰值**: 动作场面的最高潮，战斗最激烈的部分。
        * **重大情节反转**: 悬疑揭秘的关键时刻 (重要提示：为避免剧透，不得选用揭示最终结局的画面)。
    2.  **情感峰值时刻**:
        * **强烈情感爆发**: 如爱情片中主角的甜蜜热恋、激烈争吵、心碎分手等场景。
        * **悲情或震撼时刻**: 如战争片中体现战争残酷、牺牲等引发观众强烈共鸣的片段。
    3.  **视听标志性元素**:
        * **视觉奇观**: 场面宏伟、特效集中或有独特运镜的片段。
        * **听觉燃点**: 背景音乐激昂、使用经典配乐的关键段落。

    # 技术规格与内容要求 (必须严格遵守):
    1.  **对话限制**: 避免选取连续对话超过 **3秒** 的片段。
    2.  **片段时长**: 挑选出来的每个片段的时长严格控制在 **3-5秒**。
    3.  **主角特写识别**: 如果片段包含主角 ({actor_description}) 的清晰面部特写镜头，请明确标注。
    4.  **演职员表排除**: 如果片段是片尾的演职员表（滚动字幕、卡片等），请明确标注。
    5.  **标准标注**: 在`满足的核心标准`字段中，请填写具体的二级标准名称（例如："高潮对决/冲突峰值", "强烈情感爆发", "视觉奇观"），而不是一级分类（如"关键情节节点"）。

    ```输出结构```
    请严格按json结构输出，不要添加任何解释性文字。

    [
      {{
        "剧情": "此处为选定高光片段的详细剧情文字描述。",
        "剧情所属部分时间戳": "mm:ss.ms-mm:ss.ms",
        "原因": "简述选择此片段的原因，说明它符合哪几项高光标准。",
        "满足的核心标准": ["高潮对决/冲突峰值", "视觉奇观"],
        "是主角特写": "1",
        "是片尾演职员表": "0"
      }},
      ...
    ]
    """
  # 2 每段剧情描述片段控制在50s以内,保证剧情完整性.

  print("gene plots base prompt:" + prompt)
  generation_config_params = get_generation_config()

  for i, split_video in split_videos.items():
    video_links, files = [], []

    if MC_NEED_GCS:
      video_links = [split_video[1]]
    else:
      fileinfo = {
          "type": "video/*",
          "value": read_b64_from_file(split_video[0])
      }
      files = [fileinfo]

    print(f"video links:{str(i)}")
    print(video_links)
    tasks[i] = asyncio.create_task(
        genai_client.asyncCallapi(
            project_id=PROJECT_ID,
            location=LOCATION,
            model=model,
            files=files,
            textInputs=[prompt],
            videoLinks=video_links,
            imageLinks=[],
            generation_config_params=generation_config_params,
        )
    )

  result = {}
  for i, response_task in tasks.items():
    result[i] = await response_task
  return result


def gene_highlights_from_plots(plots: Dict, highlight_starring: str,
    highlight_duration: str):
  """Generate highlights from plots."""
  #     role_info = "用于组成一个新的高光故事线,保证故事的完整性"
  #     prompt = f"""请结合如下剧情信息，提取其中的片段，{role_info}, 时长严格控制在{highlight_duration}."""
  #
  #     output_str = """
  #
  # ```输出结构```
  # 按json结构输出
  #
  # [
  #   {
  #     "剧情": "此处为选定高光片段的详细剧情文字描述。", // 对选定高光片段的剧情进行描述。
  #     "剧情所属部分": "x-y", // 字符串，表示此高光片段来源于哪个分钟范围的剧情。例如："1", "3"。直接从输入描述"视频片段x剧情"中提取"x"。
  #     "剧情所属部分时间戳": "mm:ss.ms-mm:ss.ms" // 字符串，表示此高光片段在原始输入数据中，其所在的具体条目的完整`timestamp`值。例如，如果高光片段来源于"70-80分钟剧情"中`{ "timestamp": "02:21.126-03:02.885", "剧情": "马蒂尔达流着泪..." }`这一条，则此值为 "02:21.126-03:02.885"
  #     "原因":""#选择此片段的原因说明,和上下片段之间的衔接关系
  #   },
  #   ...
  # ]
  #
  # """
  #
  #     prompt += output_str
  #     prompt += """
  # You are working as part of an AI system, so no chit-chat and no explaining what you're doing and why.
  # DO NOT start with "Okay", or "Alright" or any preambles. Just the output, please."""
  # prompt = """
  #   你是一位资深的电影剪辑师和预告片制作人。用于组成一个新的高光故事线,保证故事的完整性
  #   请结合如下剧情信息，提取其中的片段， 时长严格控制在15-30秒.
  #   在挑选片段时，你必须严格遵守以下两大准则：
  #
  #   # 核心判断标准 (请综合以下几点进行艺术判断):
  #   1.  **关键情节节点**:
  #       * **主线任务触发**: 主角遇到重大挑战、与核心反派首次发生关键冲突等。
  #       * **高潮对决/冲突峰值**: 动作场面的最高潮，战斗最激烈的部分。
  #       * **重大情节反转**: 悬疑揭秘的关键时刻 (重要提示：为避免剧透，不得选用揭示最终结局的画面)。
  #   2.  **情感峰值时刻**:
  #       * **强烈情感爆发**: 如爱情片中主角的甜蜜热恋、激烈争吵、心碎分手等场景。
  #       * **悲情或震撼时刻**: 如战争片中体现战争残酷、牺牲等引发观众强烈共鸣的片段。
  #   3.  **视听标志性元素**:
  #       * **视觉奇观**: 场面宏伟、特效集中或有独特运镜的片段。
  #       * **听觉燃点**: 背景音乐激昂、使用经典配乐的关键段落。
  #
  #   # 技术规格要求 (必须严格遵守):
  #   1.  **时长**: 最终高清视频的总时长严格控制在 **15到30秒** 之间。
  #   2.  **开场要求**: 第一个被挑选的片段 **前5秒** 必须包含主角的特写镜头，或是核心冲突的直接体现，并且能清晰地揭示电影的类型（如动作、爱情、科幻）
  #   3.  **对话限制**: 避免选取连续对话超过 **10秒** 的片段。
  #
  #   你将收到按时间顺序排列的电影剧情分段描述。请基于以上所有标准，进行专业挑选。
  #
  #   ```输出结构```
  #   按json结构输出
  #
  #   [
  #     {
  #   "剧情": "此处为选定高光片段的详细剧情文字描述。",
  #       "剧情所属部分": "x-y", // 字符串，表示此高光片段来源于哪个分钟范围的剧情。例如："1", "3"。直接从输入描述"视频片段x剧情"中提取"x"。
  #       "剧情所属部分时间戳": "mm:ss.ms-mm:ss.ms", // 字符串，表示此高光片段在原始输入数据中，其所在的具体条目的完整`timestamp`值。
  #       "原因": "简述选择此片段的原因，说明它符合哪几项高光标准，并解释它在整个高光故事线中的衔接作用。" // 对选择原因进行详细说明，体现剪辑师的专业判断。
  #     },
  #     ...
  #   ]
  #
  #   You are working as part of an AI system, so no chit-chat and no explaining what you're doing and why.
  #   DO NOT start with "Okay", or "Alright" or any preambles. Just the output, please.
  #   """
  minutes = int(PARTITION_SECONDS / 60)

  for i, res in plots.items():
    part = f"{i * minutes}-{(i + 1) * minutes}"
    part = i

    is_valid, error_msg = check_response_validity(res)
    if not is_valid:
      set_error_and_return(error_msg)
      return False

    text = res[0].text
    if not text:
      print("the plot is empty")
      print(res)
      continue

    prompt += f"\n\n视频片段{part}剧情：\n{text}"""
  generation_config_params = get_generation_config(temperature=0)
  print(prompt)
  return genai_client.callapi(
      project_id=PROJECT_ID,
      location=LOCATION,
      model=st.session_state[f"model_{SUFFIX}"],
      files=[],
      textInputs=[prompt],
      videoLinks=[],
      imageLinks=[],
      generation_config_params=generation_config_params,
  )


def validate_inputs(upload_file, video_link, role_set=None) -> bool:
  """Validate input parameters."""
  if not upload_file and not video_link:
    set_error_and_return("请提供视频文件或视频链接")
    return False
  if role_set is not None and not role_set:
    set_error_and_return("请提供角色设定")
    return False
  return True


def callback_video_analysis(upload_file, video_link: str, role_set: str,
    narrative_style: str):
  """Callback for video analysis."""
  if not validate_inputs(upload_file, video_link, role_set):
    return

  st.session_state.video_first_person = {}
  st.session_state.final_output_mixed_audio_video = None
  res = narrative_video(upload_file, video_link, role_set, narrative_style)
  st.session_state[f"movie_highlight_results_{SUFFIX}"] = res

  is_valid, error_msg = check_response_validity(res)
  if not is_valid:
    set_error_and_return(error_msg)
    return

  parts = res[0].candidates[0].content.parts
  text = ""
  for part in parts:
    if part.text:
      text = part.text

  if text:
    st.session_state.video_first_person = genai_client.convert_markdown_to_json(
      text)


def process_file_input(video_link: str, uploaded_file) -> Tuple[
  Optional[str], Optional[str], Optional[str]]:
  """Process file input and return file paths and prefixes."""
  if video_link:
    file_ori_md5 = calculate_md5(video_link)
    file_prefix = f"video_{file_ori_md5}"
    file_ori_local_path = os.path.join(DATA_PATH, f"{file_prefix}.mp4")

    if os.path.exists(file_ori_local_path):
      print(
        f"step1 file already download from gcs {video_link} to {file_ori_local_path}")
    elif cloud_gcs.download_gcs_object(video_link, file_ori_local_path):
      print(
        f"step1 download file from gcs {video_link} to {file_ori_local_path}")
    else:
      set_error_and_return(
        f"下载文件失败: {video_link} 到 {file_ori_local_path}")
      return None, None, None

  elif uploaded_file:
    file_ori_md5 = calculate_md5(uploaded_file.getvalue())
    file_prefix = f"video_{file_ori_md5}"
    file_ori_local_path = os.path.join(DATA_PATH, f"{file_prefix}.mp4")

    if not write_bytes_tofile(uploaded_file, file_ori_local_path):
      set_error_and_return("save file to local fail")
      return None, None, None
  else:
    return None, None, None

  return file_ori_md5, file_prefix, file_ori_local_path


def callback_gen_highlight(video_link: str, uploaded_file):
  """Callback for Step 1: Extract Plots."""
  st.session_state.process_error = ""
  st.session_state[f"movie_plots_{SUFFIX}"] = None

  if not validate_inputs(uploaded_file, video_link):
      return

  file_ori_md5, file_prefix, file_ori_local_path = process_file_input(video_link, uploaded_file)
  if not all([file_ori_md5, file_prefix, file_ori_local_path]):
      return
  st.session_state["file_ori_local_path"] = file_ori_local_path
  st.session_state["file_prefix"] = file_prefix

  print("Step1: Splitting video for plot extraction...")
  split_videos = video_process.split_video_single_command(
      file_ori_local_path, BUCKET_NAME, OBJECT_PREFIX, PARTITION_SECONDS, MC_NEED_GCS
  )
  if not split_videos:
      set_error_and_return("视频分割失败，请重试。")
      return
  st.session_state[f"split_videos{SUFFIX}"] = split_videos
  print("Step1: Video splitting complete.")

  print("Step1: Generating video plots...")
  video_plots = gene_plots_by_splitvideos(split_videos, st.session_state.get(f"actors_info_{SUFFIX}", ""))
  
  has_error = False
  if isinstance(video_plots, dict):
      for i, r in video_plots.items():
          is_valid, error_msg = check_response_validity(r)
          if not is_valid:
              set_error_and_return(f"提取剧情时出错 (片段 {i}): {error_msg}")
              has_error = True
              break
      if not has_error:
          st.session_state[f"movie_plots_{SUFFIX}"] = video_plots
          print("Plot extraction successful.")
  else:
      set_error_and_return(f"提取剧情时发生未知错误: {video_plots}")

def callback_generate_tags():
    """Callback for the standalone 'Generate Tags' button."""
    st.session_state.process_error = ""
    st.session_state.tagging_results = None
    st.session_state.tagging_explanations = None

    gcs_video_uri = st.session_state.get("tagging_gcs_video_uri")
    
    if not gcs_video_uri:
        set_error_and_return("请为打标签功能提供独立的GCS视频地址。")
        return

    # Note: For tagging, we now only use the GCS URI. Local file upload is tied to the highlight workflow.
    # If local file support is needed for tagging, a separate uploader would be required.
    file_ori_md5, file_prefix, file_ori_local_path = process_file_input(gcs_video_uri, None)
    if not all([file_ori_md5, file_prefix, file_ori_local_path]):
        return

    if not gcs_video_uri and uploaded_file:
        gcs_object_name = f"{OBJECT_PREFIX}/{file_prefix}.mp4"
        if cloud_gcs.upload_gcs_object(BUCKET_NAME, gcs_object_name, file_ori_local_path):
            gcs_video_uri = f"gs://{BUCKET_NAME}/{gcs_object_name}"
            st.session_state[f"video_gcs_link{SUFFIX}"] = gcs_video_uri
            print(f"Uploaded local file to GCS for tagging: {gcs_video_uri}")
        else:
            set_error_and_return("本地文件上传至GCS失败，无法进行视频打标签。")
            return

    if not gcs_video_uri:
        set_error_and_return("无法获取视频的GCS地址。")
        return

    language_map = {
        "中文": "The entire output, including all keys and string values in the JSON, must be in Simplified Chinese. The predefined tags from the lists must be translated to Chinese.",
        "English": "The entire output, including all keys and string values in the JSON, must be in English.",
        "Português": "The entire output, including all keys and string values in the JSON, must be in Portuguese. The predefined tags from the lists must be translated to Portuguese."
    }
    selected_language = st.session_state.get("tagging_language", "English")  # Default to English
    language_instruction = language_map.get(selected_language, language_map["English"])

    prompt_template = """
      # ROLE
     You are a senior Film & TV Content Analyst and a Data Structuring Expert.

      TASK
      Your task is to deeply analyze the provided movie/show information and, by strictly following the "Two-Layer Multi-Dimensional Tagging System" defined below, generate a clean, logically coherent, and structurally clear JSON output.

      TWO-LAYER MULTI-DIMENSIONAL TAGGING SYSTEM
      Layer 1: Core Genres
      Rule: Choose 1-3 of the most central genres from the list below.

      List: [{lay1_core_genres}]

      Layer 2: Multi-Dimensional Keywords
      Rule: From the seven dimensional pools below, select a total of 3-7 of the most fitting descriptive tags.

      Dimension A (Theme & Core): Coming-of-Age, Revenge & Forgiveness, Love & Sacrifice, Family & Belonging, Social Justice & Injustice, Good vs. Evil, Survival & Challenge, Technology & Ethics, Power Struggle, Identity, Dreams vs. Reality, Redemption & Fall, Greed & Corruption, Loyalty & Betrayal, Fate vs. Free Will, Moral Dilemma, Class Conflict, Conspiracy, Second Chance, Dystopia
      Dimension B (Mood & Tone): Uplifting, Heartwarming, Exciting, Humorous, Lighthearted, Romantic, Hopeful, Dark, Bleak, Tearjerker, Scary, Desperate, Tense, Unsettling, Cynical, Nostalgic, Melancholy, Absurd, Dreamy, Whimsical, Suspenseful, Thought-provoking, Ironic, Satirical, Bittersweet, Offbeat, Noir
      Dimension C (Character Archetype): Anti-hero, Female Protagonist, Underdog, Genius Protagonist, Morally Ambiguous Character, The Innocent, Mentor/Sage, Rebel/Outlaw, Femme Fatale, Everyman, Charming Villain, The Chosen One, Hardboiled Detective, Mad Scientist
      Dimension D (Setting): 1980s, World War II, Cold War, Ancient Times, Medieval, Victorian Era, Roaring Twenties, Contemporary, Near Future, Distant Future, Fictional Era, Metropolis, Quiet Town, Rural/Pastoral, Campus/School, Workplace, Isolated Island, At Sea, Outer Space/Space Station, Alien Planet, Alternate Dimension/Parallel Universe, Virtual Reality, Confined Space (e.g., secret room, submarine), Post-apocalypse
      Dimension E (Narrative & Structure): Non-linear Narrative, Ensemble Cast, Found Footage, Flashback/Flashforward, Narration-driven, Stream of Consciousness, Twist Ending, Slow Pacing/Slow Burn, Fast-paced, Road Movie, Anthology, Dialogue-driven, Time Loop, Character Study
      Dimension F (Visual & Auditory Style): Visually Striking, Minimalist, Black & White, Stylized, Handheld Camera, Saturated Colors, Muted Tones, Gritty, CGI-heavy, Practical Effects-focused, Long Takes, Fast Editing, Epic Score, Electronic/Synth Score, Soundtrack-driven, Atmospheric Sound Design
      Dimension G (Source & Core Elements): Based on Novel, Based on Comic, Based on True Story, Based on Game, Robots/AI, Magic/Superpowers, Monsters/Aliens/Demons/Vampires/Zombies, Treasure Hunt, Time Travel, Political Conspiracy, Undercover, Courtroom, Heist, Martial Arts, Survival

      CONSTRAINTS & RULES
      **Language Rule**: {language_instruction}
      Strictly adhere to the "Layer 1" and "Layer 2" rules defined above.
      You MUST choose tags strictly from the provided lists; do not create new tags.
      Logical Coherence Mandate: The combination of selected tags must be logically sound. Avoid selecting tags that are mutually exclusive or create inherent contradictions.
      **Absolute Rule on Source Material**: A work CANNOT be tagged as both "Based on True Story" and "Based on Novel" simultaneously under any circumstances. You must choose only one, the most primary and direct source. This is a critical logical constraint.
      Example 2 (Mood): Avoid choosing diametrically opposed moods like "Uplifting" and "Bleak" to describe the overall tone of the same work. The tags must reflect the dominant, overarching feeling. A "Bittersweet" tag might be more appropriate for a nuanced mood.
      Example 3 (Genre): While genres can blend, avoid selecting core genres in Layer 1 that are fundamentally at odds for a mainstream piece, such as "Comedy" and "Tragedy" (or its tonal equivalent like "Tearjerker"), unless the work is a clear and deliberate hybrid like a tragicomedy.
      The output must be a pure JSON object without any additional explanations, introductions, or dialogue.
      INPUT INFORMATION
      [Please paste the specific movie/show information here, e.g., title, synopsis, director, year, etc.]
      OUTPUT FORMAT
      Please strictly return the result in the following JSON format. If no suitable tags are found for a dimension, leave its value as an empty array [].
      """
    final_prompt = prompt_template.format(
        lay1_core_genres=st.session_state.selected_genres,
        language_instruction=language_instruction
    )
    response_schema = {
        "type": "OBJECT", "properties": { "core_genres": {"type": "ARRAY", "items": {"type": "STRING"}}, "multi_dimensional_keywords": {"type": "OBJECT", "properties": { "theme_core": {"type": "ARRAY", "items": {"type": "STRING"}}, "mood_tone": {"type": "ARRAY", "items": {"type": "STRING"}}, "character_archetype": {"type": "ARRAY", "items": {"type": "STRING"}}, "setting": {"type": "ARRAY", "items": {"type": "STRING"}}, "narrative_structure": {"type": "ARRAY", "items": {"type": "STRING"}}, "visual_auditory_style": {"type": "ARRAY", "items": {"type": "STRING"}}, "source_core_elements": {"type": "ARRAY", "items": {"type": "STRING"}}, }}}}
    
    model_id = st.session_state.get(f"model_{SUFFIX}", "gemini-2.5-pro")
    
    video_uris_for_tagging = gcs_video_uri
    if st.session_state.tagging_enable_segmenting:
        print("Segmenting video for tagging...")
        split_videos = video_process.split_video_single_command(
            file_ori_local_path, BUCKET_NAME, f"{OBJECT_PREFIX}/tags", 600, True # 10-min chunks
        )
        if not split_videos:
            set_error_and_return("为打标签而分割长视频时失败。")
            return
        video_uris_for_tagging = [val[1] for val in split_videos.values() if val[1]]
        print(f"Segmented into {len(video_uris_for_tagging)} parts for tagging.")

    result = genai_client.generate_tags_for_video(
        project_id=PROJECT_ID, location=LOCATION, model_id=model_id,
        gcs_video_uri=video_uris_for_tagging, prompt=final_prompt, response_schema=response_schema,
        enable_segmenting=st.session_state.tagging_enable_segmenting
    )
    print(result)
    is_valid, error_msg = check_response_validity(result)
    if not is_valid:
        set_error_and_return(str(error_msg))
        return

    # First stage result
    initial_tags_json_str = result[0].text
    print("Tag generation successful. Now validating...")

    # --- Second Stage: Validation and Correction ---
    validation_model_id = "gemini-2.5-pro" # Use a faster, cheaper model for validation
    selected_language = st.session_state.get("tagging_language", "English")
    validation_result = genai_client.validate_and_correct_tags(
        project_id=PROJECT_ID, location=LOCATION, model_id=validation_model_id,
        generated_json_str=initial_tags_json_str,
        language=selected_language
    )
    
    is_valid_after_correction, error_msg_correction = check_response_validity(validation_result)
    if not is_valid_after_correction:
        set_error_and_return(f"标签验证和修正时出错: {error_msg_correction}")
        # Fallback to using the original result if validation fails
        st.session_state.tagging_results = genai_client.convert_markdown_to_json(initial_tags_json_str)
    else:
        # Use the corrected tags
        corrected_tags_json_str = validation_result[0].text
        st.session_state.tagging_results = genai_client.convert_markdown_to_json(corrected_tags_json_str)
        print("Tag validation and correction successful.")


# def callback_gene_highlight_by_plots(highlight_starring: str,
#     highlight_duration: str):
#   """Generate highlight by plots callback."""
#   st.session_state.process_error = ""
#
#   if not st.session_state.get(
#       "file_ori_local_path") or not st.session_state.get("file_prefix"):
#     set_error_and_return("请先执行`提取剧情`")
#     return
#
#   if not st.session_state.get(f"movie_plots_{SUFFIX}"):
#     set_error_and_return("请先执行`提取剧情`")
#     return
#
#   video_plots = st.session_state[f"movie_plots_{SUFFIX}"]
#   file_ori_local_path = st.session_state["file_ori_local_path"]
#   file_prefix = st.session_state["file_prefix"]
#
#   # Step4: Generate highlight moments from plots
#   print("Step4: start generate highlight moments from plots")
#   r = gene_highlights_from_plots(video_plots, highlight_starring,
#                                  highlight_duration)
#   if not r:
#     return
#   print(r)
#   is_valid, error_msg = check_response_validity(r)
#   if not is_valid:
#     set_error_and_return(error_msg)
#     return
#
#   print("highlight result")
#   print(r)
#
#   st.session_state[f"movie_highlight_results_{SUFFIX}"] = r
#   print("Step4: finish generate highlight moments from plots")
#
#   # Step5: Generate highlight video
#   data = genai_client.convert_markdown_to_json(r[0].text)
#   if isinstance(data, str):
#     set_error_and_return("convert to json fail" + str(r[0]))
#     return
#
#   file_highlight_video_path = os.path.join(DATA_PATH,
#                                            f"{file_prefix}_highlight.mp4")
#   gene_highlight_video(data, st.session_state[f"split_videos{SUFFIX}"],
#                        file_highlight_video_path)


def _timestamp_to_seconds(ts: str) -> float:
  """Converts a 'mm:ss.ms' timestamp string to seconds."""
  try:
    ts = ts.strip()
    parts = ts.split(':')
    minutes = int(parts[0])
    seconds_parts = parts[1].split('.')
    seconds = int(seconds_parts[0])
    milliseconds = int(seconds_parts[1])
    return minutes * 60 + seconds + milliseconds / 1000.0
  except (ValueError, IndexError) as e:
    print(f"Error converting timestamp '{ts}': {e}")
    return 0.0


def callback_gene_highlight_by_plots(highlight_starring: str,
    highlight_duration: str):
  """Generate highlight by plots callback."""
  st.session_state.process_error = ""

  if not st.session_state.get(
      "file_ori_local_path") or not st.session_state.get("file_prefix"):
    set_error_and_return("请先执行`提取剧情`")
    return

  if not st.session_state.get(f"movie_plots_{SUFFIX}"):
    set_error_and_return("请先执行`提取剧情`")
    return

  video_plots = st.session_state[f"movie_plots_{SUFFIX}"]
  file_prefix = st.session_state["file_prefix"]

  # Step4&5: Directly generate highlight video from plots
  print("Step4&5: start generate highlight video from plots")

  all_highlights = []
  # video_plots is a dict like {0: response, 1: response}
  for part_index, plot_response in video_plots.items():
    is_valid, error_msg = check_response_validity(plot_response)
    if not is_valid:
      set_error_and_return(
        f"剧情提取部分 {part_index} 返回了无效结果: {error_msg}")
      return

    text = plot_response[0].text
    if not text:
      print(f"Part {part_index} plot is empty, skipping.")
      continue

    # --- Robust JSON parsing ---
    highlight_data = []
    # Regex to find JSON-like objects, even if the list is malformed
    objects = re.findall(r'\{[^{}]*\}', text)
    for obj_str in objects:
        try:
            # Use regex for each field to be resilient to typos in structure
            plot = re.search(r'"剧情"\s*:\s*"([^"]*)"', obj_str).group(1)
            timestamp = re.search(r'"剧情所属部分时间戳"\s*:\s*"([^"]*)"', obj_str).group(1)
            reason = re.search(r'"原因"\s*:\s*"([^"]*)"', obj_str).group(1)
            
            # Handle potential typos in keys from LLM, like "特D写"
            is_closeup_match = re.search(r'"是主角特D?写"\s*:\s*"([01])"', obj_str)
            is_closeup = is_closeup_match.group(1) if is_closeup_match else "0"

            is_credits_match = re.search(r'"是片尾演职员表"\s*:\s*"([01])"', obj_str)
            is_credits = is_credits_match.group(1) if is_credits_match else "0"

            # Extract the list of met standards
            standards_match = re.search(r'"满足的核心标准"\s*:\s*(\[[^\]]*\])',
                                        obj_str)
            standards = []
            if standards_match:
              standards_str = standards_match.group(1)
              standards = [
                  s.strip().strip('"')
                  for s in standards_str.strip('[]').split(',')
                  if s.strip()
              ]

            highlight_data.append({
                "剧情": plot,
                "剧情所属部分时间戳": timestamp,
                "原因": reason,
                "满足的核心标准": standards,
                "是主角特写": is_closeup,
                "是片尾演职员表": is_credits,
            })
        except AttributeError:
            print(f"Could not parse object due to missing field, skipping: {obj_str}")
            continue
    
    if not highlight_data:
        set_error_and_return(f"从剧情片段 {part_index} 提取JSON失败: AI返回的格式不正确或内容为空。")
        return
    # --- End of robust parsing ---

    for item in highlight_data:
      item["剧情所属部分"] = part_index
    all_highlights.extend(highlight_data)

  if not all_highlights:
    set_error_and_return("未能从剧情中提取任何高光片段。")
    return

  # --- New logic for sorting and filtering ---
  # 1. Filter out credits
  no_credits_highlights = [
      item for item in all_highlights
      if item.get("是片尾演职员表") != "1"
  ]
  print(f"Removed credits, {len(no_credits_highlights)} clips remaining.")

  if not no_credits_highlights:
    set_error_and_return("所有提取的片段都是演职员表，无法生成视频。")
    return

  # 2. Find one close-up for the opening, then shuffle the rest for variety
  close_ups = [
      item for item in no_credits_highlights if item.get("是主角特写") == "1"
  ]
  other_clips = [
      item for item in no_credits_highlights if item.get("是主角特写") != "1"
  ]

  opening_shot = []
  # The pool of remaining clips starts with non-close-ups
  remaining_clips = other_clips

  if close_ups:
    # Secure one close-up for the opening
    opening_shot = [close_ups.pop(0)]
    # Add the rest of the close-ups back to the main pool
    remaining_clips.extend(close_ups)
    print(
      f"Found {len(opening_shot) + len(close_ups)} close-ups. Using one for the opening and shuffling the rest.")
  else:
    print("Warning: No protagonist close-up shot found for the opening.")

  # Sort the remaining clips by the number of criteria met (quality score)
  remaining_clips.sort(
      key=lambda item: len(item.get("满足的核心标准", [])), reverse=True)
  print(
      f"Sorted remaining {len(remaining_clips)} clips by quality score.")

  # The final list starts with the prioritized opening shot, followed by the sorted best clips
  sorted_highlights = opening_shot + remaining_clips

  # 3. Filter highlights to fit the desired duration
  print(f"Filtering highlights to fit {highlight_duration} duration.")
  filtered_highlights = []
  total_duration = 0.0
  
  # Parse duration from string like "15-30秒" or "20秒"
  duration_numbers = []
  if isinstance(highlight_duration, str):
    duration_numbers = re.findall(r'\d+', highlight_duration)

  if len(duration_numbers) == 2:
      MIN_DURATION = int(duration_numbers[0])
      MAX_DURATION = int(duration_numbers[1])
  elif len(duration_numbers) == 1:
      # If one number is given, allow a small range around it.
      val = int(duration_numbers[0])
      MIN_DURATION = max(0, val - 5)
      MAX_DURATION = val + 5
  else:
      # Default values if parsing fails
      MIN_DURATION = 15
      MAX_DURATION = 30

  for item in sorted_highlights:
    try:
      timestamp_str = item.get("剧情所属部分时间戳", "")
      if '-' not in timestamp_str:
        print(f"Skipping item due to invalid timestamp format: {item}")
        continue

      start_str, end_str = timestamp_str.split('-', 1)
      duration = _timestamp_to_seconds(end_str) - _timestamp_to_seconds(
        start_str)

      if duration <= 0:
        print(f"Skipping item due to non-positive duration: {item}")
        continue

      if total_duration + duration <= MAX_DURATION:
        filtered_highlights.append(item)
        total_duration += duration
    except Exception as e:
      print(f"Error processing timestamp for item {item}: {e}")
      continue

  if not filtered_highlights:
    set_error_and_return("无法筛选出符合时长的片段，请检查'提取剧情'步骤的输出。")
    return

  if not any(item.get("是主角特写") == "1" for item in filtered_highlights[:1]):
    print("Warning: No protagonist close-up shot found for the opening.")

  if total_duration < MIN_DURATION:
    print(
      f"Warning: Total duration is {total_duration:.2f}s, which is less than the desired {MIN_DURATION}s.")

  print(
    f"Filtered down to {len(filtered_highlights)} clips with a total duration of {total_duration:.2f}s.")

  # Store the filtered results for debugging
  st.session_state[
    f"movie_highlight_results_{SUFFIX}"] = filtered_highlights
  print("Combined and filtered highlight segments:")
  print(filtered_highlights)

  # Generate the final video
  file_highlight_video_path = os.path.join(DATA_PATH,
                                           f"{file_prefix}_highlight.mp4")
  gene_highlight_video(filtered_highlights,
                       st.session_state[f"split_videos{SUFFIX}"],
                       file_highlight_video_path)

  print("Step4&5: finish generate highlight video")


def gene_highlight_video(data: List[Dict], split_videos: Dict,
    file_highlight_video_path: str) -> bool:
  """Generate highlight video from data."""
  segments_data = []

  for item in data:
    original_timestamp = item["剧情所属部分时间戳"]
    part = item["剧情所属部分"]

    timestamps = original_timestamp.split(
      ',') if "," in original_timestamp else [original_timestamp]

    for timestamp in timestamps:
      print(timestamp)
      # new_timestamp = video_process.convert_timestamp(timestamp.strip(), part)
      segments_data.append({
          "剧情": item["剧情"],
          "选取理由": item["原因"],
          "满足的核心标准": item.get("满足的核心标准", []),
          "part": part,
          "timestamp_relative": timestamp.strip()
      })

  st.session_state[f"movie_highlight_segments_{SUFFIX}"] = segments_data
  print("segments info:")
  print(segments_data)
  # video_process.create_stitched_video_from_data(
  #     segments_data=segments_data,
  #     input_video_path=file_ori_local_path,
  #     output_video_path=file_highlight_video_path,
  #     ffmpeg_path=FFMPEG_PATH
  # )

  video_process.create_stitched_video_from_data_v2(
      segments_data,
      split_videos,
      file_highlight_video_path,
      FFMPEG_PATH
  )
  st.session_state[f"movie_highligh_video_{SUFFIX}"] = file_highlight_video_path
  return True


def prepare_video_for_gemini(file_path: str) -> Tuple[List, List]:
  """Prepare video files or links for Gemini API."""
  files, video_links = [], []
  upload_file = {
      "type": "video/mp4",
      "value": read_b64_from_file(file_path)
  }

  if MC_NEED_GCS:
    print("use gcs for gemini request")
    object_md5 = calculate_md5(upload_file["value"])
    object_name = f"{OBJECT_PREFIX}/{object_md5}.mp4"
    metadata = cloud_gcs.blob_metadata(BUCKET_NAME, object_name)

    print("start upload to gcs")
    if not metadata:
      print("Uploading to GCS...")
      cloud_gcs.upload_gcs_object(BUCKET_NAME, object_name, file_path)
    else:
      print(f"video already uploaded to gcs: {BUCKET_NAME}/{object_name}")
    video_links = [f"gs://{BUCKET_NAME}/{object_name}"]
  else:
    print("use raw bytes for gemini request")
    files = [upload_file]

  return files, video_links


def gene_highlight_selfnarration(narration_role: str, narration_style: str,
    voice_key: str, file_path: str):
  """Generate self narration for highlight."""
  narration_language = "Should output by chinese"
  if "en_" in voice_key:
    narration_language = "Should output by english"

  prompt = f"""
```role```
你是一位经验丰富的电影编剧，精通人物心理剖析与角色重塑。你擅长解读角色的潜在动机、内心挣扎，并能精准把握和调整人物的语调。你的任务是分析提供的视频，按照 "角色设定", 创作该角色视角的第一人称叙事，并与视频内容完美同步。

```角色设定```
{narration_role}

```instructions```
1. **深入分析视频内容**：逐帧或逐场景地理解视频中的视觉元素（人物、动作、表情、环境、物品、镜头语言等）、听觉元素（对话、音效、背景音乐氛围等）以及它们共同传达的剧情、情感和节奏。
2. **智能切分视频片段**：根据内容和叙事节奏，将视频划分为若干个逻辑连贯的小片段，并为每个片段确定精确的 `timestamp` (mm:ss-mm:ss) 范围。
3. **生成旁白**：为每个切分出的视频片段，根据下面的"旁白风格指令"创作旁白。
4. **确保音画同步**：为每一句旁白精确设置 `旁白开始时间`，确保其与视频画面内容紧密对应。
5. **输出指定JSON格式**：将所有旁白条目整合到最终的JSON数组中。
6. 语言要生动、具有感染力，能够引导观众沉浸到你的视角中，仿佛与你一同经历电影中的故事。
7. 不要直接重复电影中的台词。如果需要提及对话，请用你自己的方式转述，并侧重于该对话对你造成的影响或你的内心反应。


#旁白风格指令 (请在生成时严格遵循)：
1.以严格第一人称视角撰写内心独白, 独白的开头要进行自我介绍,独白风格参考**{narration_style}风格**。
**在整个叙事的开篇（通常是第一个视频片段，或前几个片段的组合，如果开场画面切换很快），用一两句话自然地融入自我介绍**，点出"我是谁（乌鸦，卧底警察），我正在做什么/我此刻的感受"。这个介绍**不应该**生硬地独立成一个极短的旁白条目，而是要巧妙地结合对开场画面的观察或内心感受。例如，可以将自我介绍作为对第一个有意义的场景旁白的开头部分。
**此后的所有旁白则专注于视频内容解读、情节推进和角色在当前情境下的内心活动，无需重复自我介绍。
**独白应与视频内容匹配, 但内容不要和视频中内容重复。独白内容要深刻体现 "角色设定" 的个性和当时的处境。
2.叙事与描述：
**画面解读**: 精准描述当前画面中的核心人物、动作、表情和环境细节。
**情节推进**: 清晰地叙述故事发展，连接不同场景和事件。
**背景/心理**: 适时补充角色背景、内心活动或状态。

3.节奏与同步:
a 旁白应紧跟视频节奏，一句或几句旁白对应一小段具体的视觉内容。
b narration_relative_time 的设置至关重要，通常应在相关视觉信息或动作**开始呈现后的0-2秒内**启动旁白，以实现最佳的音画同步效果。

4.语言风格:
a 口语化、自然流畅，易于理解。
b 句子不宜过长，多用短句。
c 根据情节发展，语气可以有平缓叙述、略带悬念、适当强调或轻微感叹。
d 符合中国人叙事风格
e 避免过于书面化的解读，可以改为更具情感、更含蓄或更有引导性


```output```
请输出JSON格式
[
 {{
 "timestamp":"mm:ss-mm:ss", // AI智能切分出的视频片段的时间戳范围 , 时间戳范围不要重叠，避免旁白出现交叉
 "narration_relative_time":"", // 旁白相对视频片段的相对时间,【重要】整数, 为相对 "timestamp" 的开始时间, 以s为单位, 如5代表5s。此值通常较小（如0, 1, 2），以确保旁白与画面内容紧密同步。
 "narration": "" // 根据上述风格生成的旁白文本,尽量控制在30字以内, {narration_language}
 }},
 ... 
]
"""

  files, video_links = prepare_video_for_gemini(file_path)

  return genai_client.callapi(
      project_id=PROJECT_ID,
      location=LOCATION,
      model=st.session_state[f"model_{SUFFIX}"],
      files=files,
      textInputs=[prompt],
      videoLinks=video_links,
      imageLinks=[],
      generation_config_params=get_generation_config(temperature=0),
      is_async=False
  )


def gene_highlight_narration(narration_character_nickname: str, voice_key: str,
    file_path: str):
  """Generate highlight narration."""
  narration_language = "Should output by chinese"
  if "en_" in voice_key:
    narration_language = "Should output by english"

  movie_bg = st.session_state[f"narration_movie_bg_{SUFFIX}"]

  prompt = f"""
#角色
你是一位专业的视频解说员和内容分析师，擅长深度观看和精准分析视频，并能创作出引人入胜、音画高度同步的旁白。你将用清晰、简洁且富有情感或悬念的语言，描述画面、解读人物行为与心理，并适时补充背景信息，引导观众沉浸式理解剧情。

#任务
请分析提供的视频,{movie_bg},按以下步骤操作创作旁白：
1. **深度视频分析**：逐帧或逐场景解析视频的视觉元素（人物、动作、表情、环境、物品、镜头语言等）和听觉元素（对话、音效、背景音乐氛围等），理解它们共同传达的剧情、情感和叙事节奏。
2. **智能片段切分**：依据内容关联性和叙事节奏，将视频划分为若干逻辑连贯的小片段，并为每个片段确定精确的 `timestamp` (mm:ss-mm:ss) 范围，确保时间戳范围不重叠。
3. **旁白创作**：选取适合做旁白的视频片段，遵循下述"旁白风格与内容要求"创作旁白。
4. **音画精准同步**：为每句旁白精确设置 `narration_relative_time` (旁白相对片段的开始时间，单位：秒，如0, 1, 2)，确保旁白与视频画面内容紧密对应，通常在相关视觉信息出现后的0-2秒内启动。
5. **指定JSON输出**：将所有旁白条目整合到最终的JSON数组中。

# 旁白风格与内容要求 (请在生成时严格遵循)：

## 核心原则
1. **风格定位**：参考优秀电影解说、高燃混剪解说风格，采用符合中国人叙事习惯的口语化、自然流畅的语言。
2. **超越画面**：避免对画面的简单描述或台词的直接复述，融入独到见解、背景补充或深层含义解读，引导观众思考。
3. **引人入胜**：开场和转场部分需设置引题，语言略带悬念或情感色彩，激发观众好奇心。

## 语言表达
1. **生动形象**：多用动词和形象化的词语，善用恰当的比喻和习语，让文案"活"起来，增强画面感和感染力。
  * 例如动词："摔翻"、"割裂"、"撞见"、"威逼利诱"、"迅速后退"、"瞬间中刀"、"愤怒暴走"。
  * 例如比喻/习语："火药味越发浓烈"、"三下五除二"、"话不投机半句多"。
2. **口语与书面语结合**：主体采用口语化表达增加亲和力，可适度融入概括性书面语，使文案既生动又不失格调。
3. **句式简练**：多用短句，句子不宜过长，便于观众快速理解和接收。
4. **语气多变**：根据情节发展，语气可灵活调整为平缓叙述、略带悬念、适当强调或轻微感叹。
5. **避免过度解读**：避免过于书面化、学术化的解读。
  * 例如，避免："镜头特写，小美脸颊上的淤青清晰可见，暗示着她不幸的家庭生活。"
  * 可改为更具情感或引导性："脸颊上的这块淤青，是小美无法言说的秘密，也是她家庭生活投下的阴影。"
6**旁白文案**: 请参考王家卫电影台词风格。

## 叙事节奏与结构
1. **逻辑清晰**：提前梳理剧情骨架，确保解说逻辑连贯，有效引导观众理解剧情。
2. **转场自然**：善用"与此同时"、"片刻"、"这下"、"当即"、"此后"等转场词，保持叙事的连贯性和多线叙事的流畅性。
3. **节奏把控**：根据影片内容（文戏或武戏）调整语速和句子长度，做到张弛有度。
4. **悬念与冲突**：在关键转折点或危机来临前，通过铺垫制造紧张气氛；通过对话和行为解读展现人物间的智谋交锋与心理博弈。
5. **适时总结**：在适当节点加入总结性旁白，概括当前剧情核心进展，帮助观众梳理脉络。
{"6. **角色称呼**：" + narration_character_nickname if narration_character_nickname else ""}。

# 输出格式 (JSON)
[
 {{ "timestamp":"mm:ss-mm:ss", // AI智能切分出的视频片段的时间戳范围 , 时间戳范围不要重叠，避免旁白出现交叉
 "narration_relative_time":"", // 旁白相对视频片段的相对时间,【重要】整数, 为相对 "timestamp" 的开始时间, 以s为单位, 如5代表5s。此值通常较小（如0, 1, 2），以确保旁白与画面内容紧密同步。
 "narration": "" // 根据上述风格生成的旁白文本  {narration_language}
 }},
 ...
]
"""

  files, video_links = prepare_video_for_gemini(file_path)

  return genai_client.callapi(
      project_id=PROJECT_ID,
      location=LOCATION,
      model=st.session_state[f"model_{SUFFIX}"],
      files=files,
      textInputs=[prompt],
      videoLinks=video_links,
      imageLinks=[],
      generation_config_params=get_generation_config(temperature=0),
      is_async=False
  )


def validate_voice_and_video(voice_key: str, highlight_video_path: str) -> bool:
  """Validate voice key and highlight video path."""
  if not voice_key:
    set_error_and_return("请先选择配音")
    return False
  if not highlight_video_path:
    set_error_and_return("请先生成高光视频")
    return False
  return True


def callback_gen_highlight_narration(highlight_video_path: str,
    narration_role_set: str,
    narration_style: str, narration_character_nickname: str,
    voice_key: str, narration_type_selected: str):
  """Generate highlight narration callback."""
  st.session_state.process_error = ""

  if not validate_voice_and_video(voice_key, highlight_video_path):
    return

  # Generate narration based on type
  if narration_type_selected == "第一人称叙事":
    r = gene_highlight_selfnarration(narration_role_set, narration_style,
                                     voice_key, highlight_video_path)
  else:
    r = gene_highlight_narration(narration_character_nickname, voice_key,
                                 highlight_video_path)

  print(r)
  is_valid, error_msg = check_response_validity(r)
  if not is_valid:
    st.session_state[f"movie_narration_{SUFFIX}"] = r
  else:
    r_json = genai_client.convert_markdown_to_json(r[0].text)
    st.session_state[f"movie_narration_{SUFFIX}"] = r_json


def callback_gene_narration_video(highlight_video_path: str, voice_key: str):
  """Generate narration video callback."""
  st.session_state.process_error = ""

  if not validate_voice_and_video(voice_key, highlight_video_path):
    return

  data = st.session_state[f"movie_narration_{SUFFIX}"]

  # Fix potential key naming issues
  for v in data:
    v["narration_relative_time"] = v.get("narration_relative_time",
                                         v.get("narration_relative_time\n", 0))

  # Generate synthetic audio
  print("callback_gen_highlight_narration: start synthetic audio")
  narration, narration_files = synthetic_audio(SPEECH_PROJECT_ID, data,
                                               voice_key)

  # Generate output path
  base, _ = os.path.splitext(highlight_video_path)
  final_output_mixed_audio_video = f"{base}_narration.mp4"

  # Mix audio with video
  video_process.mix_narrations_with_video_audio(
      stitched_video_path=highlight_video_path,
      segments_data=narration,
      narration_audio_files_dict=narration_files,
      final_output_path=final_output_mixed_audio_video,
      main_audio_normal_volume=3,
      main_audio_ducked_volume=0.2,
      narration_volume=3,
      ffmpeg_path=FFMPEG_PATH,
      ffprobe_path=FFPROBE_PATH,
      segments_data_type="stitched"
  )
  st.session_state[
    f"movie_highligh_video_narration_{SUFFIX}"] = final_output_mixed_audio_video


def callback_voice_synthesis(upload_file, video_link: str, voice_key: str):
  """Voice synthesis callback."""
  if not voice_key:
    set_error_and_return("请先选择配音")
    return
  if not st.session_state.get("video_first_person"):
    set_error_and_return("请先提取视频精彩片段")
    return
  if not validate_inputs(upload_file, video_link):
    return

  data = st.session_state.video_first_person
  gcs_link = video_link

  # Download original file
  file_ori_md5 = calculate_md5(gcs_link)
  file_prefix = f"video_{file_ori_md5}"
  file_ori_local_path = os.path.join(DATA_PATH, f"{file_prefix}.mp4")

  if os.path.exists(file_ori_local_path):
    print(
      f"step1 file already download from gcs {gcs_link} to {file_ori_local_path}")
  elif cloud_gcs.download_gcs_object(gcs_link, file_ori_local_path):
    print(f"step1 download file from gcs {gcs_link} to {file_ori_local_path}")
  else:
    set_error_and_return(f"下载文件失败: {gcs_link} 到 {file_ori_local_path}")
    return

  # Create merged video
  file_dest_md5 = calculate_md5(str(data))
  file_dest_path = os.path.join(DATA_PATH,
                                f"video_mix_{file_ori_md5}_{file_dest_md5}.mp4")

  if not os.path.exists(file_dest_path):
    print(f"将从 '{file_ori_local_path}' 提取片段并合并到 '{file_dest_path}'")
    video_process.create_stitched_video_from_data(data, file_ori_local_path,
                                                  file_dest_path,
                                                  ffmpeg_path=FFMPEG_PATH)
  st.session_state.mid_video = file_dest_path

  # Generate synthetic audio
  print("Voice synthesis initiated with selected voice.")
  narration_files = synthetic_audio(SPEECH_PROJECT_ID, data, voice_key)
  st.session_state["audio_file_path"] = narration_files
  print(st.session_state["audio_file_path"])

  # Mix audio with video
  final_output_mixed_audio_video = os.path.join(DATA_PATH,
                                                "final_video_with_all_narrations.mp4")
  r = video_process.mix_narrations_with_video_audio(
      stitched_video_path=file_dest_path,
      segments_data=data,
      narration_audio_files_dict=narration_files,
      final_output_path=final_output_mixed_audio_video,
      main_audio_normal_volume=1,
      main_audio_ducked_volume=0.2,
      narration_volume=3.0,
      ffmpeg_path=FFMPEG_PATH,
      ffprobe_path=FFPROBE_PATH
  )

  if r:
    st.session_state[
      "final_output_mixed_audio_video"] = final_output_mixed_audio_video


def narrative_video(upload_file, video_link: str, role_set: str,
    narrative_style: str):
  """Generate narrative video."""
  prompt = f"""```role```
你是一位经验丰富的电影编剧，精通人物心理剖析与角色重塑。你擅长解读角色的潜在动机、内心挣扎，并能精准把握和调整人物的语调。你的任务是分析提供的视频，按照 "角色设定" 选取相关的精彩视频片段,创建出一个新的6-8分钟故事线视频,同时创作第一人称叙事，并与视觉效果完美同步。

```角色设定```
{role_set}

```instructions```
1.准确选取和 "角色设定人物" 有关的精彩视频片段,每个片段时间控制在30s,同时保证视频片段剧情的完整性. **结合视频和你的经验进行审查，保证选取的视频片段中是和 "角色设定人物" 有关，不要提供错误的片段**
2.确保精彩片段分布在整个视频中，从开头到中间到结尾，而不是集中在前几分钟。
3.以第一人称撰写内心独白, 独白的开头要进行自我介绍,独白风格参考**{narrative_style}风格**。独白应与所选的精彩视频片段直接对应,不要和视频中内容重复。
4.请筛选合适的片段用于结尾，不要加入任何独白。
5.独白开始时间是相对每个选取的视频片段的独白适合开始出现的时间,以s为单位,指示它应在何时在其相应的视频片段内开始独白，以得到提升视频展示效果，如视频播放的某某内容，独白和该内容呼应。
""" + """#output
[
{
"timestamp":"mm:ss-mm:ss",
"narration_relative_time":"", #如果不存在,请返回空值。
"narration": "" #如果不存在,请返回空值
},
...
]
"""

  video_links, files = [], []
  if video_link:
    video_links = [video_link]
  elif upload_file:
    files = [upload_file]

  print(video_links)
  print(prompt)

  return genai_client.callapi(
      project_id=PROJECT_ID,
      location=LOCATION,
      model=st.session_state[f"model_{SUFFIX}"],
      files=files,
      textInputs=[prompt],
      videoLinks=video_links,
      imageLinks=[],
      generation_config_params=get_generation_config(temperature=0.01),
      is_async=False
  )


def synthetic_audio(project_id: str, input_data: List[Dict], voice_key: str) -> \
Tuple[List[Dict], Dict[int, str]]:
  """Synthesize audio from narration data."""
  voice = VOICES[voice_key]
  lang = "en-US" if "en_" in voice_key else "cmn-CN"

  narration = []
  ret = {}
  i = 0

  for item in input_data:
    text = item.get("narration")

    if not text:
      print("narration内容为空，跳过合成。")
      continue

    print(f"开始合成音频: {text}")
    md5 = calculate_md5(str(text))
    audio_file_path = os.path.join(DATA_PATH, f"audio_{voice_key}_{md5}.wav")

    if os.path.exists(audio_file_path):
      print(f"音频文件已存在: {audio_file_path}")
    elif "minimax_" in voice_key:
      genai_speech.callMiniMax(text, voice, audio_file_path, 1.16, 0, 1.5)
      time.sleep(3)
    elif "google_" in voice_key:
      genai_speech.synthesize_text_with_hd_voice(text, voice, audio_file_path,
                                                 lang, 1.4)
    else:
      genai_speech.synthesize_text_with_cloned_voice(project_id, voice, text,
                                                     audio_file_path, lang)

    narration.append(item.copy())
    ret[i] = audio_file_path
    i += 1
    print("音频合成完成")

  return narration, ret


def reset():
  """Reset process error."""
  st.session_state.process_error = ""


def example(color1: str, color2: str, color3: str, content: str):
  """Display styled example text."""
  st.markdown(
      f'<p style="text-align:left;background-image: linear-gradient(to right,{color1}, {color2});color:{color3};">{content}</p>',
      unsafe_allow_html=True
  )


def upload_callback():
  """Handle file upload callback."""
  print("upload_file on change")
  upload_key = f"upload_file_{SUFFIX}_{st.session_state.uploader_key}"
  print(st.session_state.get(upload_key))

  if st.session_state.get(upload_key):
    st.session_state[f"upload_{SUFFIX}"] = st.session_state[upload_key]
    st.session_state[f"video_gcs_link{SUFFIX}"] = ""
  print(st.session_state[f"upload_{SUFFIX}"])


def callback_gcs_link():
  """Handle GCS link callback."""
  print("video_gcs_link on change")
  if st.session_state[f"video_gcs_link{SUFFIX}"]:
    st.session_state[f"upload_{SUFFIX}"] = None
    st.session_state.uploader_key += 1


# Initialize Streamlit
if "init" not in st.session_state:
  streamlit_init()

# Sidebar
with st.sidebar:
  sidebar_tips = """
操作步骤:
\n1 提取剧情
\n2 生成高光时刻视频
\n
"""
  st.sidebar.success(sidebar_tips)
  st.selectbox("model", MODEL_LIST, key=f"model_{SUFFIX}")
  st.selectbox("debug", [False, True], key="mc_debug")

  st.button("reset")

# Main content
with st.container(border=0):
  example(COLOR_PARAMS, COLOR_PARAMS_GRAD, COLOR_TEXT, "视频标签生成")
  with st.expander("视频标签参数设定", expanded=True):
    # Define constants from Colab
    LAYER1_MOVIE_CORE_GENRES = "Action, Adult, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Family, Fantasy, Film Noir, Game Show, History, Horror, Musical, Music, Mystery, News, Reality-TV, Romance, Sci-Fi, Short, Sport, Talk-Show, Thriller, War, Western, Kids"
    LAYER1_TV_CORE_GENRES = "Anime, Business & Finance, Classic TV, Comedy, Crime & Mystery, Documentary, Educational, Entertainment, Faith & Spirituality, Gaming, Geek, Health & Wellness, Horror, International, Kids & Family, Lifestyle, Local, Movies, Music, Nature, News, Reality TV, Sci-Fi & Fantasy, Soap Operas, Sports, Travel, TV Shows, Variety, Westerns"

    st.text_input("GCS视频地址 (仅用于打标签)", key="tagging_gcs_video_uri")
    
    tagging_cols = st.columns(3)
    with tagging_cols[0]:
        content_type = st.selectbox("内容类型", ["Movie", "TV"], key="tagging_content_type")
    with tagging_cols[1]:
        st.selectbox("输出语言", ["中文", "English", "Português"], key="tagging_language")
    with tagging_cols[2]:
        enable_segmenting = st.checkbox("为长视频启用分段处理", value=True, key="tagging_enable_segmenting")

    if content_type == "Movie":
        st.session_state.selected_genres = LAYER1_MOVIE_CORE_GENRES
    else:
        st.session_state.selected_genres = LAYER1_TV_CORE_GENRES
    st.button("生成视频标签", on_click=callback_generate_tags)

  if st.session_state.get("tagging_results"):
    with st.expander("视频标签结果", expanded=True):
        st.json(st.session_state.tagging_results)

  example(COLOR_RESULTS, COLOR_RESULTS_GRAD, COLOR_TEXT, "高光视频生成")
  with st.expander("高光视频参数设定", expanded=False):
    # Pre-defined video list with full GCS paths
    PRESET_VIDEOS = {
        "A_Hora_do_Acerto.mp4": "gs://tcl-tv-video/watchnow/A_Hora_do_Acerto.mp4",
        "Arn_O_Cavaleiro_Templário.mp4": "gs://tcl-tv-video/watchnow/Arn_Cavaleiro_Templário.mp4",
        "O_Duelo_Final.mp4": "gs://tcl-tv-video/watchnow/Duelo_Final.mp4",
        "Hacker_Todo_Crime_Tem_um_Inicio.mp4": "gs://tcl-tv-video/watchnow/Hacker_Todo_Crime_Tem_um_Inicio.mp4",
        "O_Milagre.mp4": "gs://tcl-tv-video/watchnow/Milagre.mp4",
        "Redbad_A_Invasão_dos_Francos.mp4": "gs://tcl-tv-video/watchnow/Redbad_A_Invasão_dos_Francos.mp4",
        "Teia_de_Corrupção.mp4": "gs://tcl-tv-video/watchnow/Teia_de_Corrupção.mp4",
        "Tesouro_Perdido.mp4": "gs://tcl-tv-video/watchnow/Tesouro_Perdido.mp4",
        "Três_Ladrões_e_um_Bebê.mp4": "gs://tcl-tv-video/watchnow/Três_Ladrões_e_um_Bebê.mp4",
        "Um_Anjo_em_Nossas_Vidas.mp4": "gs://tcl-tv-video/watchnow/Um_Anjo_em_Nossas_Vidas.mp4"
    }
    MANUAL_INPUT_OPTION = "手动输入 GCS 地址..."
    
    video_options = [MANUAL_INPUT_OPTION] + list(PRESET_VIDEOS.keys())
    selected_option = st.selectbox("选择或输入GCS视频", options=video_options, key="video_selector")

    if selected_option == MANUAL_INPUT_OPTION:
        st.text_input("请在此处粘贴完整的GCS地址", key=f"video_gcs_link{SUFFIX}", on_change=callback_gcs_link)
    else:
        gcs_video_uri = PRESET_VIDEOS[selected_option]
        st.session_state[f"video_gcs_link{SUFFIX}"] = gcs_video_uri
        # Use a different key for the info text to avoid conflicts
        st.info(f"已选择: {gcs_video_uri}")

    cols_highlight = st.columns(3)
    with cols_highlight[0]:
      st.text_input("演员信息(辅助人物准确识别)", value="",
                    key=f"actors_info_{SUFFIX}")
    with cols_highlight[1]:
      st.text_input("高光短片主角", value="",
                    key=f"highlight_starring_{SUFFIX}")
    with cols_highlight[2]:
      st.text_input("高光短片时长",
                    key=f"highlight_duration_{SUFFIX}")

    st.file_uploader(
        "上传本地文件 (如果选择, 将覆盖GCS地址)",
        key=f"upload_file_{SUFFIX}_{st.session_state.uploader_key}",
        type=["mp4", "mkv", "webm"],
        on_change=upload_callback
    )



  # with st.expander("旁白配音参数设定"):
  #   cols_character = st.columns(4)
  #   with cols_character[0]:
  #     st.text_input("角色设定",
  #                   value="将张秋生（由李保田饰演）重新设想为一个超级英雄",
  #                   key=f"role_set_{SUFFIX}")
  #   with cols_character[1]:
  #     st.text_input("叙述风格", value="王家卫电影",
  #                   key=f"narration_style_{SUFFIX}")
  #   with cols_character[2]:
  #     st.text_input("角色代号", value="李保田称之为老李",
  #                   key=f"narration_character_nickname_{SUFFIX}")
  #   with cols_character[3]:
  #     st.text_input("影片背景",
  #                   value="视频为《有话好好说》关于角色张秋生的剪辑视频，请参考此背景以及人物的设定",
  #                   key=f"narration_movie_bg_{SUFFIX}")
  #
  #   cols_voice = st.columns(4)
  #   with cols_voice[0]:
  #     st.selectbox("选择旁白类型", ["第三人称叙事", "第一人称叙事"],
  #                  key=f"narration_type_selected")
  #   with cols_voice[1]:
  #     st.selectbox("选择配音", VOICES.keys(), index=2, key=f"voice_selected")
  #   with cols_voice[2]:
  #     st.text_input("minimax_group_id", key="minimax_group_id")
  #   with cols_voice[3]:
  #     st.text_input("minimax_api_key", key="minimax_api_key")

  example(COLOR_RESULTS, COLOR_RESULTS_GRAD, COLOR_TEXT, "操作步骤:")
  cols_button_mc = st.columns(4)
  with cols_button_mc[0]:
    st.button(
        "Step1: 提取剧情",
        on_click=callback_gen_highlight,
        args=(st.session_state.get(f"video_gcs_link{SUFFIX}"),
              st.session_state.get(f"upload_{SUFFIX}"))
    )
  with cols_button_mc[1]:
    st.button(
        "Step2: 生成高光视频",
        on_click=callback_gene_highlight_by_plots,
        args=(st.session_state[f"highlight_starring_{SUFFIX}"],
              st.session_state[f"highlight_duration_{SUFFIX}"])
    )
  # with cols_button_mc[2]:
  #   st.button(
  #       "Step3: 生成视频旁白",
  #       on_click=callback_gen_highlight_narration,
  #       args=(
  #           st.session_state[f"movie_highligh_video_{SUFFIX}"],
  #           st.session_state[f"role_set_{SUFFIX}"],
  #           st.session_state[f"narration_style_{SUFFIX}"],
  #           st.session_state[f"narration_character_nickname_{SUFFIX}"],
  #           st.session_state[f"voice_selected"],
  #           st.session_state[f"narration_type_selected"]
  #       )
  #   )
  # with cols_button_mc[3]:
  #   st.button(
  #       "Step4: 合成旁白视频",
  #       on_click=callback_gene_narration_video,
  #       args=(st.session_state[f"movie_highligh_video_{SUFFIX}"],
  #             st.session_state[f"voice_selected"])
  #   )
  # example(COLOR_RESULTS, COLOR_RESULTS_GRAD, COLOR_TEXT, "-----")

  st.write("")

  if st.session_state.process_error:
    st.write("oops, 处理异常了，请参考如下输出进行处理 or Ask gemini cli:")
    st.warning(st.session_state.process_error)

  with st.container(border=0):
    example(COLOR_OPERATIONS, COLOR_OPERATIONS_GRAD, COLOR_TEXT,
            "高光视频输出")
    cols = st.columns(2)
    with cols[0]:
      with st.container(border=1, height=500):
        if st.session_state.get(f"movie_highlight_segments_{SUFFIX}"):
          st.write("movie_highlight_segments:")
          st.write(st.session_state[f"movie_highlight_segments_{SUFFIX}"])
        if st.session_state.get(f"movie_plots_{SUFFIX}"):
          st.write("movie_plots:")
          for k, v in st.session_state[f"movie_plots_{SUFFIX}"].items():
            st.write(v[0].text)
    with cols[1]:
      if st.session_state.get(f"movie_highligh_video_{SUFFIX}"):
        st.write("高光时刻视频")
        st.video(st.session_state[f"movie_highligh_video_{SUFFIX}"])

  # with st.container(border=0):
  #   example(COLOR_OPERATIONS, COLOR_OPERATIONS_GRAD, COLOR_TEXT,
  #           "旁白配音相关输出")
  #   cols_narration = st.columns(2)
  #   with cols_narration[0]:
  #     if st.session_state.get(f"movie_narration_{SUFFIX}"):
  #       narration_data = st.session_state[f"movie_narration_{SUFFIX}"]
  #       if isinstance(narration_data, list) and len(narration_data) > 3 and \
  #           narration_data[3] == "fail":
  #         st.write(narration_data)
  #       else:
  #         with st.container(border=1, height=500):
  #           if isinstance(narration_data, list):
  #             for i, value in enumerate(narration_data):
  #               if isinstance(value, dict):
  #                 relative_time = value.get("narration_relative_time",
  #                                           value.get(
  #                                             "narration_relative_time\n", 0))
  #
  #                 st.session_state[f"movie_narration_{SUFFIX}"][i][
  #                   "narration"] = st.text_area(
  #                     f"视频时间 {value.get('timestamp', '')} 旁白开始时间:{relative_time}s",
  #                     value.get("narration", "")
  #                 )
  #           st.write(st.session_state[f"movie_narration_{SUFFIX}"])
  #
  #   with cols_narration[1]:
  #     if st.session_state.get(f"movie_highligh_video_narration_{SUFFIX}"):
  #       st.video(st.session_state[f"movie_highligh_video_narration_{SUFFIX}"])

# Debug section
if st.session_state.get("mc_debug"):
  example(COLOR_DEBUG, COLOR_DEBUG_GRAD, COLOR_TEXT, "debug相关输出:")
  st.write("debug data:")
  if st.session_state.get(
      f"movie_highlight_segments_{SUFFIX}") or st.session_state.get(
      f"movie_plots_{SUFFIX}"):
    with st.container(border=1, height=500):
      if st.session_state.get(f"movie_highlight_segments_{SUFFIX}"):
        st.write("movie_highlight_segments:")
        st.write(st.session_state[f"movie_highlight_segments_{SUFFIX}"])
      if st.session_state.get(f"movie_plots_{SUFFIX}"):
        st.write("movie_plots:")
        for k, v in st.session_state[f"movie_plots_{SUFFIX}"].items():
          st.write(v)

  if st.session_state.get(f"movie_highlight_results_{SUFFIX}"):
    st.write("movie_highlight_results:")
    with st.container(border=1, height=500):
      st.write(st.session_state[f"movie_highlight_results_{SUFFIX}"])
