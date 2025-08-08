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


import subprocess
import shlex # For safely quoting arguments if needed
from ailib import cloud_gcs
import subprocess
import os
import shlex
import json # 需要导入 json 来解析 ffprobe 输出
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import subprocess
import shutil
from pathlib import Path
import glob
from typing import List, Dict

def get_video_duration_ffmpeg(video_path: str) -> float | None:
    """
    使用 ffprobe (通过 subprocess) 获取视频的时长（秒）。

    参数:
        video_path (str): 视频文件的路径。

    返回:
        float: 视频时长（秒），如果无法获取则返回 None。
    """
    # 构建 ffprobe 命令
    # -v error: 只显示错误信息
    # -show_format: 显示容器格式信息，其中包含时长
    # -show_streams: 显示流信息，某些情况下格式时长可能缺失，可以从流中获取
    # -of json: 以 JSON 格式输出，方便解析
    # -print_format json: 另一种指定JSON输出的方式，兼容性更好
    command = [
        "ffprobe",
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-of", "json", # 或者使用 "-print_format", "json",
        video_path
    ]

    try:
        # 执行命令
        # capture_output=True 捕获标准输出和标准错误
        # text=True (或 universal_newlines=True) 将输出解码为文本
        # check=True 如果命令返回非零退出码，则抛出 CalledProcessError
        process_result = subprocess.run(
            command,
            capture_output=True,
            text=True, # 或者 universal_newlines=True
            check=True, # 如果 ffprobe 失败则抛出异常
            encoding='utf-8' # 明确指定编码
        )

        # 解析 JSON 输出
        metadata = json.loads(process_result.stdout)

        print(metadata)

        # 尝试从 format -> duration 获取时长
        if 'format' in metadata and 'duration' in metadata['format']:
            duration_str = metadata['format']['duration']
            return float(duration_str)
        
        # 如果 format 中没有时长，尝试从 streams 中获取 (通常是视频或音频流)
        # 有时视频可能没有全局的 'format' 时长，但流本身有
        if 'streams' in metadata:
            for stream in metadata['streams']:
                if 'duration' in stream:
                    # 选择第一个找到的流时长，或者可以进一步判断选择视频流还是音频流
                    duration_str = stream['duration']
                    # 有些流时长可能为 "N/A"
                    if duration_str != "N/A":
                        return float(duration_str)
        
        print(f"警告: 在 '{video_path}' 的元数据中未找到明确的 'duration' 字段。")
        return None

    except subprocess.CalledProcessError as e:
        print(f"错误: ffprobe 执行失败。")
        print(f"命令: {' '.join(e.cmd)}")
        print(f"返回码: {e.returncode}")
        print(f"错误输出: {e.stderr}")
        return None
    except FileNotFoundError:
        print("错误: ffprobe 命令未找到。请确保 FFmpeg (包含 ffprobe) 已安装并在系统 PATH 中。")
        return None
    except json.JSONDecodeError:
        print(f"错误: 解析 ffprobe 的 JSON 输出失败。路径: '{video_path}'")
        return None
    except ValueError as e:
        print(f"错误: 无法将获取到的时长转换为数字。时长字符串可能无效。错误: {e}")
        return None
    except Exception as e:
        print(f"获取视频 '{video_path}' 时长时发生未知错误: {e}")
        return None

def time_to_seconds(time_str):
    """Converts M:SS or S or M:S.ms string to seconds (float)."""
    if not time_str:
        raise ValueError("Time string cannot be empty or None")
    parts = str(time_str).split(':')
    if len(parts) == 3:
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours*3600 + minutes * 60 + seconds
        except ValueError:
            raise ValueError(f"Invalid time format in M:S component: {time_str}")
    elif len(parts) == 2:
        try:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        except ValueError:
            raise ValueError(f"Invalid time format in M:S component: {time_str}")
    elif len(parts) == 1:
        try:
            return float(parts[0])
        except ValueError:
            raise ValueError(f"Invalid time format for seconds: {time_str}")
    else:
        raise ValueError(f"Invalid time format: {time_str}. Expected M:S or S.")

def parse_timestamp_range(timestamp_str):
    """Parses a timestamp range like 'M1:S1-M2:S2' or 'S1-S2' into start and end seconds."""
    if not timestamp_str or '-' not in timestamp_str:
        raise ValueError(f"Invalid timestamp range format: {timestamp_str}. Expected 'START-END'.")
    start_str, end_str = timestamp_str.split('-', 1)
    start_seconds = time_to_seconds(start_str)
    end_seconds = time_to_seconds(end_str)
    if end_seconds <= start_seconds:
        raise ValueError(f"End time must be after start time in range: {timestamp_str}")
    return start_seconds, end_seconds

def create_stitched_video_from_data(segments_data, input_video_path, output_video_path, ffmpeg_path='ffmpeg'):
    """
    Creates a new video by stitching segments from an input video using FFmpeg.

    Args:
        segments_data (list): A list of dictionaries, each with a 'timestamp' key.
        input_video_path (str): Path to the source video file.
        output_video_path (str): Path for the newly created output video file.
        ffmpeg_path (str): Path to the FFmpeg executable.
    """
    filter_complex_parts = []
    # These lists will now correctly store just the labels for concat
    video_stream_labels_for_concat = []
    audio_stream_labels_for_concat = []
    valid_segment_count = 0 # Use this for 'n' in concat and for unique labels

    for i, segment_info in enumerate(segments_data):
        timestamp_str = segment_info.get("timestamp")
        
        if not timestamp_str:
            print(f"Info: Skipping segment {i+1} as 'timestamp' is missing or null.")
            continue
            
        try:
            start_s, end_s = parse_timestamp_range(timestamp_str)
        except ValueError as e:
            print(f"Warning: Could not parse timestamp '{timestamp_str}' for segment {i+1}. Error: {e}. Skipping this segment.")
            continue

        video_label = f"v{valid_segment_count}" # e.g., v0, v1, v2
        audio_label = f"a{valid_segment_count}" # e.g., a0, a1, a2

        # Trim video and audio for the current segment
        filter_complex_parts.append(f"[0:v]trim=start={start_s:.3f}:end={end_s:.3f},setpts=PTS-STARTPTS[{video_label}];")
        filter_complex_parts.append(f"[0:a]atrim=start={start_s:.3f}:end={end_s:.3f},asetpts=PTS-STARTPTS[{audio_label}];")
        
        video_stream_labels_for_concat.append(f"[{video_label}]")
        audio_stream_labels_for_concat.append(f"[{audio_label}]")
        valid_segment_count += 1

    if not valid_segment_count: # If no valid segments were processed
        print("Error: No valid video segments found to process. Output video will not be created.")
        return

    # Correctly build the input stream labels part for the concat filter
    # e.g., [v0][a0][v1][a1][v2][a2]
    concat_inputs_str = ""
    for i in range(valid_segment_count):
        concat_inputs_str += video_stream_labels_for_concat[i] + audio_stream_labels_for_concat[i]
    
    # Build the concat filter string
    concat_filter_string = concat_inputs_str + \
                           f"concat=n={valid_segment_count}:v=1:a=1[outv][outa]"
    
    full_filter_complex = "".join(filter_complex_parts) + concat_filter_string
    print(full_filter_complex)
    ffmpeg_command = [
        ffmpeg_path,
        '-i', input_video_path,
        '-filter_complex', full_filter_complex,
        '-map', '[outv]',
        '-map', '[outa]',
        # Optional: Add encoding parameters here for better quality/size control
        '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
        '-c:a', 'aac', '-b:a', '192k',
        output_video_path,
        '-y' 
    ]

    print("Generated FFmpeg command:")
    print(" ".join(shlex.quote(arg) for arg in ffmpeg_command))

    try:
        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("\nFFmpeg process completed successfully.")
        print(f"Output video saved to: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nError during FFmpeg execution (return code {e.returncode}):")
        # print("FFmpeg Stdout:", e.stdout) # Can be very verbose
        print("FFmpeg Stderr:", e.stderr)
    except FileNotFoundError:
        print(f"\nError: FFmpeg executable not found at '{ffmpeg_path}'. "
              "Please ensure FFmpeg is installed and the path is correct.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


import subprocess
import shlex
import json # For ffprobe output parsing
import os

# --- Utility Functions ---


def get_media_duration(media_file_path, ffprobe_path='ffprobe'):
    """Gets the duration of an audio or video file using ffprobe."""
    if not os.path.exists(media_file_path):
        raise FileNotFoundError(f"Media file not found: {media_file_path}")
        
    command = [
        ffprobe_path,
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        media_file_path
    ]
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        return float(process.stdout.strip())
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe error getting duration for {media_file_path}: {e.stderr}")
    except FileNotFoundError:
        raise FileNotFoundError(f"ffprobe executable not found at '{ffprobe_path}'. Please ensure ffprobe (part of FFmpeg) is installed and the path is correct.")
    except ValueError:
        raise RuntimeError(f"Could not parse duration from ffprobe output for {media_file_path}.")

# --- Main Function ---

def mix_narrations_with_video_audio(
    stitched_video_path, 
    segments_data, 
    narration_audio_files_dict, 
    final_output_path,
    main_audio_normal_volume=0.7, # Example: original audio at 70% volume normally
    main_audio_ducked_volume=0.1, # Example: original audio at 10% volume during narration
    narration_volume=2,         # Example: narration at 150% volume (can be > 1 for boost)
    ffmpeg_path='ffmpeg',
    ffprobe_path='ffprobe',
    segments_data_type = "original" # "original" or "stitched"
):
    """
    Mixes narration audio files into a stitched video, ducking original audio during narration.

    Args:
        stitched_video_path (str): Path to the video file created from stitching segments.
        segments_data (list): List of dictionaries describing the original segments
                              (used to calculate narration timing in the stitched video).
                              Each dict must have "timestamp" (for segment duration) and
                              can have "narration_relative_time" and an implicit link to a narration audio.
        narration_audio_files_dict (dict): Dictionary mapping segment index (or other unique key
                                           corresponding to segments_data) to narration audio file paths.
        final_output_path (str): Path for the final output video with mixed audio.
        main_audio_normal_volume (float): Volume of original video audio when no narration is playing.
        main_audio_ducked_volume (float): Volume of original video audio when narration is playing.
        narration_volume (float): Volume for the narration audio.
        ffmpeg_path (str): Path to FFmpeg executable.
        ffprobe_path (str): Path to ffprobe executable.
    """

    if not os.path.exists(stitched_video_path):
        print(f"Error: Stitched video file not found: {stitched_video_path}")
        return

    ffmpeg_inputs = ['-i', stitched_video_path]
    filter_complex_parts = []
    
    active_narrations_info = [] # Stores (narration_input_index, abs_start_s, abs_end_s, stream_label)
    
    current_stitched_video_time = 0.0
    narration_input_idx_counter = 1 # Starts from 1 as input 0 is the stitched video

    for i, segment_info in enumerate(segments_data):
        segment_duration_s = 0
        original_timestamp_str = segment_info.get("timestamp")

        if original_timestamp_str: # Calculate duration of this segment in stitched video
            try:
                start_s_orig, end_s_orig = parse_timestamp_range(original_timestamp_str)
                segment_duration_s = end_s_orig - start_s_orig
            except ValueError as e:
                print(f"Warning: Cannot calculate duration for segment {i} from timestamp '{original_timestamp_str}'. Assuming 0 duration for timing. Error: {e}")
                # This could lead to timing issues if not all segments have valid durations.
                # For the last segment if its timestamp is null, this might be okay if it has no narration.

        if segments_data_type == "stitched":
            current_stitched_video_time = start_s_orig

        narration_relative_start_str = segment_info.get("narration_relative_time")
        # Check if this segment has a narration associated with it in the dict
        narration_file_path = narration_audio_files_dict.get(i) # Assuming dict keys are 0-indexed segment numbers

        if narration_file_path and narration_relative_start_str is not None:
            if not os.path.exists(narration_file_path):
                print(f"Warning: Narration file for segment {i} ('{narration_file_path}') not found. Skipping narration.")
            else:
                try:
                    narration_relative_start_s = narration_relative_start_str
                    narration_duration_s = get_media_duration(narration_file_path, ffprobe_path)

                    abs_narration_start_s = current_stitched_video_time + float(narration_relative_start_s)
                    abs_narration_end_s = abs_narration_start_s + narration_duration_s

                    ffmpeg_inputs.extend(['-i', narration_file_path])
                    
                    nar_label = f"n{len(active_narrations_info)}" # e.g., n0, n1
                    filter_complex_parts.append(
                        f"[{narration_input_idx_counter}:a]volume={narration_volume},"
                        f"adelay={int(abs_narration_start_s * 1000)}|{int(abs_narration_start_s * 1000)}[{nar_label}];"
                    )
                    active_narrations_info.append({
                        "start": abs_narration_start_s,
                        "end": abs_narration_end_s,
                        "label": f"[{nar_label}]"
                    })
                    narration_input_idx_counter += 1
                except (ValueError, RuntimeError, FileNotFoundError) as e:
                    print(f"Warning: Error processing narration for segment {i} ('{narration_file_path}'). Error: {e}. Skipping narration.")
        
        if original_timestamp_str: # Only advance time if segment had a calculable duration
             current_stitched_video_time += segment_duration_s


    if not active_narrations_info:
        print("Info: No narration audio to mix. If this is unexpected, check narration file paths and 'narration_relative_time'.")
        # Optionally, one could just copy the input to output if no narrations
        # For now, we'll proceed to build command, which will effectively just re-encode audio.
        # Or, simply return if no processing is desired.
        # Let's make it just adjust the main audio volume if no narrations.

    # Ducking filter for the main audio from stitched video
    if active_narrations_info:
        ducking_conditions = "+".join([f"between(t,{info['start']:.3f},{info['end']:.3f})" for info in active_narrations_info])
        filter_complex_parts.append(
            f"[0:a]volume=enable='{ducking_conditions}:volume={main_audio_ducked_volume}'[a_ducked];"
        )
        
        # Amix filter
        amix_inputs_str = "[a_ducked]" + "".join([info['label'] for info in active_narrations_info])
        num_amix_inputs = 1 + len(active_narrations_info)
        filter_complex_parts.append(
            f"{amix_inputs_str}amix=inputs={num_amix_inputs}:duration=first:dropout_transition=0:normalize=0[a_mix]"
        )
        final_audio_map_label = "[a_mix]"
    else: # No narrations, just apply normal volume to main audio (or bypass filtering)
        filter_complex_parts.append(
            f"[0:a]volume={main_audio_normal_volume}[a_final];"
        )
        final_audio_map_label = "[a_final]"


    full_filter_complex = "".join(filter_complex_parts)

    ffmpeg_command = [ffmpeg_path] + ffmpeg_inputs
    if full_filter_complex: # Only add filter_complex if there's something to filter
        ffmpeg_command.extend(['-filter_complex', full_filter_complex])
    
    ffmpeg_command.extend([
        '-map', '0:v',            # Map video from the first input (stitched_video_path)
        '-map', final_audio_map_label, # Map the processed audio
        '-c:v', 'copy',           # Copy video stream without re-encoding (as per example)
        '-c:a', 'aac',            # Encode audio to AAC
        '-b:a', '192k',           # Audio bitrate 192k
        final_output_path,
        '-y'                      # Overwrite output if exists
    ])

    print("Generated FFmpeg command:")
    print(" ".join(shlex.quote(arg) for arg in ffmpeg_command))

    try:
        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("\nFFmpeg process completed successfully.")
        print(f"Output video saved to: {final_output_path}")
        return active_narrations_info
    except subprocess.CalledProcessError as e:
        print(f"\nError during FFmpeg execution (return code {e.returncode}):")
        # print("FFmpeg Stdout:", e.stdout)
        print("FFmpeg Stderr:", e.stderr)
    except FileNotFoundError:
        print(f"\nError: FFmpeg/ffprobe executable not found. Searched for '{ffmpeg_path}' and '{ffprobe_path}'. "
              "Please ensure FFmpeg (which includes ffprobe) is installed and the paths are correct.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

from datetime import datetime, timedelta
def convert_timestamp(timestamp_str, part_str):
    """
    根据剧情所属部分调整时间戳。

    Args:
      timestamp_str: 原始时间戳字符串，格式为 "MM:SS-MM:SS"
      part_str: 剧情所属部分字符串，例如 "0-10", "10-20", "20-30"

    Returns:
      调整后的时间戳字符串，格式为 "MM:SS-MM:SS"
    """
    start_str, end_str = timestamp_str.split('-')

    bool_include_ms = False
    if "." in start_str:
        start_time, start_ms = start_str.split('.')
        end_time, end_ms = end_str.split('.')
        start_time = datetime.strptime(start_time, "%M:%S")
        end_time = datetime.strptime(end_time, "%M:%S")
        bool_include_ms = True
    else:
        start_time = datetime.strptime(start_str, "%M:%S")
        end_time = datetime.strptime(end_str, "%M:%S")

    parts = part_str.split('-') 
    minDelta = parts[0]
    start_time += timedelta(minutes=int(minDelta),seconds=0)
    end_time += timedelta(minutes=int(minDelta),seconds=0)

    if bool_include_ms:
        return f"{start_time.strftime('%H:%M:%S')}.{start_ms}-{end_time.strftime('%H:%M:%S')}.{end_ms}"
    else:
        return f"{start_time.strftime('%H:%M:%S')}-{end_time.strftime('%H:%M:%S')}"


def format_seconds_to_hhmmss(total_seconds: float | int) -> str:
    """
    将总秒数转换为 HH:MM:SS 格式的字符串。

    参数:
        total_seconds (float | int): 要转换的总秒数。

    返回:
        str: HH:MM:SS 格式的时间字符串。
    """
    if not isinstance(total_seconds, (int, float)):
        raise TypeError("输入的总秒数必须是整数或浮点数。")
    if total_seconds < 0:
        raise ValueError("输入的总秒数不能为负。")

    # 将总秒数转换为整数，因为时间单位通常是整数
    total_seconds = int(round(total_seconds)) # 四舍五入到最近的整数秒

    hours = total_seconds // 3600
    remaining_seconds = total_seconds % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60

    # 格式化为两位数字，不足则补零
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def split_video_ffmpeg(input_video_path, output_video_path, start_time, end_time, ffmpeg_path='ffmpeg', bucket_name=None, object_path=None): 
    process_id = os.getpid()
    print(f"[{process_id}] Worker 开始处理: {output_video_path}")

    if not os.path.exists(output_video_path):
        cmd = [
            ffmpeg_path,
            '-i', input_video_path,
            '-ss', format_seconds_to_hhmmss(start_time),
            '-to', format_seconds_to_hhmmss(end_time),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', '192k',
            output_video_path,
            '-y'  # Overwrite output file if it exists
        ]
        print(f"[{process_id}]Running command:", " ".join(shlex.quote(arg) for arg in cmd))
        subprocess.run(cmd, check=True)
    else:
        print(f"Output video already exists: {output_video_path}")
    
    metadata = cloud_gcs.blob_metadata(bucket_name, object_path)
    if  not metadata:
        # upload to gcs
        print("Uploading to GCS...")
        cloud_gcs.upload_gcs_object(bucket_name, object_path, output_video_path)
    else:
        print(f"video already uploaded to gcs: {bucket_name}/{object_path}")

    return {"output_video_path": output_video_path, "gcs_uri": f"gs://{bucket_name}/{object_path}"}


def split_video_by_multiprocess(input_video, output_dir, file_prefix, partition_seconds= 600, bucket_name=None, object_path=None,max_workers=10):
    duration = get_video_duration_ffmpeg(input_video)
    if duration is None:
        print(f"无法获取视频 {input_video} 的时长。中止。")
        return {}
    
    num_segments = int(duration/partition_seconds)
    print(f"计算得到 {num_segments} 个分片。")

    os.makedirs(output_dir, exist_ok=True)

    results = {}
    futures_map = {} # 用于将 future 映射回分片索引或信息
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in range(num_segments):
            start = i * 600
            end = (i + 1) * 600
            if end > duration:
                end = duration

            output_filename = f"{output_dir}/{file_prefix}{i}.mp4"
            file_name = os.path.basename(output_filename)   
            object_path = f"{object_path}/{file_name}"

            future = executor.submit(
                    split_video_ffmpeg, # 要在子进程中执行的函数
                    input_video_path=input_video,
                    output_video_path=output_filename,
                    start_time=start,
                    end_time=end,
                    ffmpeg_path="ffmpeg",
                    bucket_name=bucket_name,
                    object_path=object_path
                )
            futures_map[future] = {"index": i, "output_path": output_filename, "gcs_path": f"gs://{bucket_name}/{object_path}"}
        for future in as_completed(futures_map):
            task_info = futures_map[future]
            segment_index = task_info["index"]
            try:
                result_data = future.result() # 获取结果，如果任务中发生异常，这里会重新抛出
                results[segment_index] = result_data
                print(f"主进程: 收到分片 {segment_index} 的结果: {result_data}")
            except Exception as e:
                print(f"主进程: 分片 {segment_index} (路径: {task_info['output_path']}) 执行时发生严重错误: {e}")
                results[segment_index] = {
                    "output_video_path": task_info['output_path'],
                    "gcs_uri": None,
                    "status": "process_execution_failed",
                    "error": str(e)
                }
    return results
   
def split_video(input_video, output_dir, file_prefix, partition_seconds= 600, bucket_name=None, object_path=None, min_last_segment_duration= 10):
    duration = get_video_duration_ffmpeg(input_video)
    if duration is None:
        print(f"无法获取视频 {input_video} 的时长。中止。")
        return {}
    
    num_segments = int(duration/partition_seconds)
    print(f"计算得到 {num_segments} 个分片。")

    remainder_seconds = duration % partition_seconds
    if remainder_seconds > min_last_segment_duration:
        num_segments += 1
        print(f"余下部分 ({remainder_seconds:.2f} 秒) 超过 {min_last_segment_duration} 秒，计为一个有效分片。总分片数增加 1。")
    elif remainder_seconds > 0 : # 如果有余数，但不满足条件
        print(f"余下部分 ({remainder_seconds:.2f} 秒) 未超过 {min_last_segment_duration} 秒，将被忽略。")
    else: # remainder_seconds == 0，即正好是 partition_seconds 的整数倍
        print(f"视频时长正好是标准分片时长的整数倍，无额外余下部分。")

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for i in range(num_segments):
        start = i * partition_seconds
        end = (i + 1) * partition_seconds
        if end > duration:
            end = duration

        output_filename = f"{output_dir}/{file_prefix}{i}.mp4"
        file_name = os.path.basename(output_filename)   

        results[i] = split_video_ffmpeg(
            input_video_path=input_video,
            output_video_path=output_filename,
            start_time=start,
            end_time=end,
            ffmpeg_path="ffmpeg",
            bucket_name=bucket_name,
            object_path= f"{object_path}/{file_name}"
        )
    return results 


def split_video_single_command(
        video_path: str, 
        bucket_name: str,
        object_path: str,
        segment_duration_sec: int, 
        need_upload_to_gcs: bool
    ) -> List[str]:
    """
    使用单一的 ffmpeg 命令和 segment 复用器将视频高效地分割成均匀长度的部分。

    这是性能最优的方法。

    :param video_path: 完整的输入视频文件路径 (例如 "/a/b/c/filename.mp4")。
    :param segment_duration_sec: 每个分段的时长（秒）。
    :return: 一个包含所有生成的文件路径的列表。
    """
    # --- 1. 检查和准备 ---
    if not shutil.which('ffmpeg'):
        print("错误: ffmpeg 命令未找到。请确保 FFmpeg 已正确安装并添加到系统 PATH。")
        return []

    path_obj = Path(video_path)
    if not path_obj.is_file():
        print(f"错误: 输入的视频路径不是一个有效文件 -> {video_path}")
        return []

    # --- 2. 路径和文件名模式处理 ---
    file_dir = path_obj.parent
    file_stem = path_obj.stem
    file_suffix = path_obj.suffix

    # 为 ffmpeg 创建输出文件名模式，例如: /a/b/c/filename_part_%d.mp4
    # %d 会被 ffmpeg 自动替换为 0, 1, 2, ...
    # 如果希望是 001, 002 这种格式，可以使用 %03d
    output_pattern = file_dir / f"{file_stem}_{segment_duration_sec}_part_%d{file_suffix}"
    
    print(f"输入视频: {path_obj}")
    print(f"分段时长: {segment_duration_sec} 秒")
    print(f"输出模式: {output_pattern}")

    # --- 3. 构建并执行单一的 ffmpeg 命令 ---
    command = [
        'ffmpeg',
        '-i', str(path_obj),              # 输入文件
        '-c', 'copy',                     # 直接复制流
        '-f', 'segment',                  # 使用 segment 复用器
        '-segment_time', str(segment_duration_sec), # 设置分段时长
        '-reset_timestamps', '1',         # 重置时间戳
        '-y',                             # 覆盖已存在的文件
        str(output_pattern)               # 指定输出模式
    ]

    print("\n正在执行高效分割命令...")
    try:
        # 执行命令
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print("✅ [成功] ffmpeg 命令执行完毕。")
    except subprocess.CalledProcessError as e:
        print(f"❌ [失败] ffmpeg 命令执行时出错。")
        print(f"FFmpeg 错误信息:\n{e.stderr.decode('utf-8', errors='ignore')}")
        return []

    # --- 4. 查找并返回生成的文件列表 ---
    # 由于是 ffmpeg 自动生成的文件，我们需要在命令执行后去查找它们
    search_glob = file_dir / f"{file_stem}_{segment_duration_sec}_part_*{file_suffix}"

    related_files = glob.glob(str(search_glob))

    created_files = sorted(
        related_files,
        key=lambda f: int(re.search(r'_part_(\d+)', f).group(1))
    )
    
    print(f"\n成功生成了 {len(created_files)} 个文件。")

    file_index_dict = {filename: index for index, filename in enumerate(created_files)}
    # print(file_index_dict)

    if need_upload_to_gcs:
        r = cloud_gcs.upload_parts_to_gcs_parallel(
            local_files = created_files, 
            bucket_name= bucket_name, 
            object_path = object_path, 
            max_workers  = 8
        )

        r_by_index_order = {}

        for filename,index in file_index_dict.items():
            r_by_index_order[index] = r[filename]
        return r_by_index_order
    else:
        # r = {file_path: None for file_path in created_files}
        r_by_index_order = {index: [filename] for index, filename in enumerate(created_files)}
    return  r_by_index_order
   

def create_stitched_video_from_data_v2(segments_data, split_videos, output_video_path, ffmpeg_path='ffmpeg'):
    """
    Creates a new video by stitching segments from split video files using FFmpeg.
    
    This is the v2 version that works with pre-split video files instead of a single original video.

    Args:
        segments_data (list): A list of dictionaries, each with:
            - 'part': string indicating which split video to use (corresponds to split_videos key)
            - 'timestamp_relative': string in format "MM:SS.ms-MM:SS.ms" indicating the relative time in that part
        split_videos (dict): Dictionary where:
            - key: part index (int) 
            - value: list where first element [0] is the local file path of the split video
        output_video_path (str): Path for the newly created output video file.
        ffmpeg_path (str): Path to the FFmpeg executable.

    Example:
        segments_data = [
            {"part": "0", "timestamp_relative": "00:00.000-00:12.583"},
            {"part": "0", "timestamp_relative": "00:12.583-00:28.416"},
            {"part": "1", "timestamp_relative": "00:05.000-00:20.000"}
        ]
        
        split_videos = {
            0: ['/path/to/part0.mp4', 'gs://bucket/part0.mp4', 'Uploaded'],
            1: ['/path/to/part1.mp4', 'gs://bucket/part1.mp4', 'Uploaded']
        }
    """
    filter_complex_parts = []
    video_stream_labels_for_concat = []
    audio_stream_labels_for_concat = []
    valid_segment_count = 0
    
    # Dictionary to track which split videos we need as inputs
    required_inputs = {}
    input_mapping = {}  # Maps part number to input index in ffmpeg command
    input_counter = 0

    # First pass: identify required input files and validate segments
    for i, segment_info in enumerate(segments_data):
        part_str = segment_info.get("part")
        timestamp_str = segment_info.get("timestamp_relative")
        
        if part_str is None:
            print(f"Warning: Segment {i+1} missing 'part' field. Skipping.")
            continue
            
        if not timestamp_str:
            print(f"Warning: Segment {i+1} missing 'timestamp_relative' field. Skipping.")
            continue
            
        try:
            part_index = int(part_str)
        except ValueError:
            print(f"Warning: Invalid part index '{part_str}' for segment {i+1}. Skipping.")
            continue
            
        if part_index not in split_videos:
            print(f"Warning: Part {part_index} not found in split_videos for segment {i+1}. Skipping.")
            continue
            
        split_video_info = split_videos[part_index]
        if not split_video_info or len(split_video_info) == 0:
            print(f"Warning: No file path found for part {part_index} in segment {i+1}. Skipping.")
            continue
            
        video_file_path = split_video_info[0]  # First element is the local file path
        if not os.path.exists(video_file_path):
            print(f"Warning: Video file '{video_file_path}' for part {part_index} not found. Skipping segment {i+1}.")
            continue
            
        # Track this input if we haven't seen it before
        if part_index not in required_inputs:
            required_inputs[part_index] = video_file_path
            input_mapping[part_index] = input_counter
            input_counter += 1

    if not required_inputs:
        print("Error: No valid video segments found to process. Output video will not be created.")
        return

    # Build the ffmpeg input list
    ffmpeg_inputs = []
    for part_index in sorted(required_inputs.keys()):
        ffmpeg_inputs.extend(['-i', required_inputs[part_index]])
    
    # Second pass: process segments and build filter complex
    for i, segment_info in enumerate(segments_data):
        part_str = segment_info.get("part")
        timestamp_str = segment_info.get("timestamp_relative")
        
        if part_str is None or not timestamp_str:
            continue
            
        try:
            part_index = int(part_str)
            if part_index not in split_videos or part_index not in input_mapping:
                continue
                
            start_s, end_s = parse_timestamp_range(timestamp_str)
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not process segment {i+1}. Error: {e}. Skipping.")
            continue

        input_index = input_mapping[part_index]
        video_label = f"v{valid_segment_count}"
        audio_label = f"a{valid_segment_count}"

        # Trim video and audio for the current segment from the appropriate input
        filter_complex_parts.append(f"[{input_index}:v]trim=start={start_s:.3f}:end={end_s:.3f},setpts=PTS-STARTPTS[{video_label}];")
        filter_complex_parts.append(f"[{input_index}:a]atrim=start={start_s:.3f}:end={end_s:.3f},asetpts=PTS-STARTPTS[{audio_label}];")
        
        video_stream_labels_for_concat.append(f"[{video_label}]")
        audio_stream_labels_for_concat.append(f"[{audio_label}]")
        valid_segment_count += 1

    if not valid_segment_count:
        print("Error: No valid video segments were processed. Output video will not be created.")
        return

    # Build the concat filter string
    concat_inputs_str = ""
    for i in range(valid_segment_count):
        concat_inputs_str += video_stream_labels_for_concat[i] + audio_stream_labels_for_concat[i]
    
    concat_filter_string = concat_inputs_str + f"concat=n={valid_segment_count}:v=1:a=1[outv][outa]"
    
    full_filter_complex = "".join(filter_complex_parts) + concat_filter_string
    print("Filter complex:")
    print(full_filter_complex)
    
    # Build the complete ffmpeg command
    ffmpeg_command = [ffmpeg_path] + ffmpeg_inputs + [
        '-filter_complex', full_filter_complex,
        '-map', '[outv]',
        '-map', '[outa]',
        '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
        '-c:a', 'aac', '-b:a', '192k',
        output_video_path,
        '-y' 
    ]

    print("Generated FFmpeg command:")
    print(" ".join(shlex.quote(arg) for arg in ffmpeg_command))

    try:
        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("\nFFmpeg process completed successfully.")
        print(f"Output video saved to: {output_video_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError during FFmpeg execution (return code {e.returncode}):")
        print("FFmpeg Stderr:", e.stderr)
        return False
    except FileNotFoundError:
        print(f"\nError: FFmpeg executable not found at '{ffmpeg_path}'. "
              "Please ensure FFmpeg is installed and the path is correct.")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return False


# r = split_video_single_command(
#     "/Users/oolongz/Desktop/project/gemini-analysis-frontend/src/data/video_8d615241875caaa91a4f33865a52fc60.mp4",  
#     "gemini-oolongz",
#     "tmp",
#     60,
#     True 
# )

# print(r)


# segments_data = [{"剧情":"故事的开篇，我们看到了李保田的故乡——湖北。清晨的云雾缭绕着江边的城市和连绵的茶山，这里山清水秀，自古便是产茶胜地。这片土地不仅养育了李保田，也孕育了他家族世代相传的茶事业。","timestamp":"00:00:00.000-00:00:12.583","选取理由":"选择此片段作为故事的开端，通过壮丽的自然风光，建立主角李保田与故乡土地的深厚联系，点明其故事根植于湖北深厚的茶文化背景之中，为后续讲述其家族传承和事业奠定环境和情感基调。","part":"0","timestamp_relative":"00:00.000-00:12.583"},{"剧情":"李保田的家族世代制茶，他们遵循着古老的技艺。镜头中，我们看到李保田和他的师傅们，围在巨大的炒锅旁，用双手感知着温度，翻炒、揉捻着茶叶。这不仅仅是制茶，更是一种流传了千年的匠心与专注，是李保田家族的立身之本。","timestamp":"00:00:12.583-00:00:28.416","选取理由":"承接开篇的环境介绍，此片段具体展示了李保田家族的核心技艺——传统手工制茶。这不仅展现了茶文化的深度，也塑造了李保田作为传承人的“匠人”身份，为后续他坚守祖业、寻求突破埋下伏笔。","part":"0","timestamp_relative":"00:12.583-00:28.416"},{"剧情":"李保田时常会走上那条祖辈们走过的古道。青苔斑驳的石阶，记录着“万里茶道”的沧桑。他仿佛能看到几百年前，他的祖先牵着马匹，将家乡的茶叶一步步运往遥远的异国。那段旅途充满了艰辛，但也开启了家族的荣耀。","timestamp":"00:01:14.033-00:01:24.533","选取理由":"将故事从当下拉回历史，通过“万里茶道”的意象，揭示李保田家族辉煌的过去。这为李保田的人物增添了历史厚重感，也解释了他内心深处那份想要重振家族荣光的使命感，是故事的核心情感驱动力。","part":"1","timestamp_relative":"00:14.033-00:24.533"},{"剧情":"这就是李保田家族世代制作的青砖茶，也是当年祖辈们行销万里的骄傲。他轻轻抚摸着这块压制紧实的茶砖，感受着其中蕴含的岁月与汗水。对于李保田来说，这块茶不仅仅是商品，更是家族历史的见证和精神的载体。","timestamp":"00:01:24.533-00:01:28.833","选取理由":"在宏大的历史叙事后，用一个特写镜头聚焦于核心物品“青砖茶”。这使得故事的载体变得具体、可感。它既是连接过去与现在的纽带，也是后续“羊来茶往”故事的关键物品，起到了承上启下的作用。","part":"1","timestamp_relative":"00:24.533-00:28.833"},{"剧情":"时光流转，历史的机遇悄然而至。为了感谢蒙古国捐赠三万只羊的情谊，国家决定回赠一批湖北的特色茶叶。李保田家族制作的赤壁青砖茶，因其深厚的历史底蕴和卓越的品质而被选中。这一刻，李保田意识到，他的茶将承载着新的使命，踏上一条新的“茶道”。","timestamp":"00:01:48.066-00:02:00.099","选取理由":"这是故事的转折点和高潮。将李保田的个人奋斗与重大的国家叙事结合起来，极大地提升了故事的格局。他家族的茶叶被选中，不仅是对其品质的认可，也让他个人的传承使命与国家间的友好交流联系在了一起，完成了人物弧光的升华。","part":"1","timestamp_relative":"00:48.066-01:00.099"},{"剧情":"最终，李保田的故事达到了顶点。“羊来茶往”的佳话，让古老的“万里茶道”在新时代焕发了新的光彩。它不再仅仅是一条商业贸易之路，更成为了一条跨越国界的友谊之路。李保田站在故乡的山水间，他知道，自己不仅继承了祖辈的技艺，更将这份承载着友谊与和平的茶香，传向了更远的地方。","timestamp":"00:02:01.800-00:02:06.500","选取理由":"作为故事的结尾，此片段用宏大的视听语言和明确的字幕，为李保田的故事赋予了崇高的意义。将“羊来茶往”定义为新时代的“万里茶道”，完美呼应了前文的历史铺垫，使李保田的个人奋斗史最终汇入时代洪流，完成了故事的闭环，留下了悠远的回味。","part":"2","timestamp_relative":"00:01.800-00:06.500"}]
# split_videos = {0: ['/Users/oolongz/Desktop/project/gemini-analysis-frontend/src/data/video_6aed61cd2819e654adbdd76d7f2e72a1_60_part_0.mp4', 'gs://gemini-oolongz/movie-demo-part/video_6aed61cd2819e654adbdd76d7f2e72a1_60_part_0.mp4', 'Uploaded'], 1: ['/Users/oolongz/Desktop/project/gemini-analysis-frontend/src/data/video_6aed61cd2819e654adbdd76d7f2e72a1_60_part_1.mp4', 'gs://gemini-oolongz/movie-demo-part/video_6aed61cd2819e654adbdd76d7f2e72a1_60_part_1.mp4', 'Uploaded'], 2: ['/Users/oolongz/Desktop/project/gemini-analysis-frontend/src/data/video_6aed61cd2819e654adbdd76d7f2e72a1_60_part_2.mp4', 'gs://gemini-oolongz/movie-demo-part/video_6aed61cd2819e654adbdd76d7f2e72a1_60_part_2.mp4', 'Uploaded']}
# output_video_path = "/tmp/output.mp4"
# create_stitched_video_from_data_v2(segments_data, split_videos, output_video_path)