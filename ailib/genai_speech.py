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

import requests, os, json, base64
#ffmpeg -i input.aifc -acodec pcm_s16le -ac 1 -ar 24000 output.wav
#http://sherlog/_1MCNt6KPaQ 
import binascii
import google.auth
import google.auth.transport.requests
from google.api_core.client_options import ClientOptions
from google.cloud import texttospeech_v1beta1 as texttospeech
from google.oauth2 import service_account
from ailib.env_config import (
    SA_KEY_FILE_PATH, 
    SPEECH_AUTH_TYPE,
    MINIMAX_GROUP_ID,
    MINIMAX_API_KEY
)
import streamlit as st

def get_speech_client():
    """
    创建通用的 Text-to-Speech 客户端
    根据 SPEECH_AUTH_TYPE 配置选择认证方式:
    - SA: 使用服务账户认证
    - 其他: 使用默认认证 (ADC)
    """
    credentials = None
    
    if SPEECH_AUTH_TYPE == "SA":
        if SA_KEY_FILE_PATH and os.path.exists(SA_KEY_FILE_PATH):
            credentials = service_account.Credentials.from_service_account_file(
                SA_KEY_FILE_PATH, 
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
        else:
            print(f"警告: SA 认证模式但服务账户文件不存在: {SA_KEY_FILE_PATH}")
            print("回退到默认认证模式")
    
    return texttospeech.TextToSpeechClient(
        client_options=ClientOptions(api_endpoint="texttospeech.googleapis.com"),
        credentials=credentials
    )


def get_token():
    """
    获取访问令牌
    根据 SPEECH_AUTH_TYPE 配置选择认证方式
    """
    if SPEECH_AUTH_TYPE == "SA" and SA_KEY_FILE_PATH and os.path.exists(SA_KEY_FILE_PATH):
        credentials = service_account.Credentials.from_service_account_file(
            SA_KEY_FILE_PATH, 
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
    else:
        credentials, _ = google.auth.default()
    
    # 刷新访问令牌
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)
    # 获取访问令牌
    access_token = credentials.token
    return access_token


def create_instant_custom_voice_key( project_id, reference_audio_bytes, consent_audio_bytes, language_code="en-US"):
    access_token = get_token()
    url = "https://texttospeech.googleapis.com/v1beta1/voices:generateVoiceCloningKey"
  
    reference_audio_b64 = base64.b64encode(reference_audio_bytes).decode("utf-8")
    consent_audio_b64 = base64.b64encode(consent_audio_bytes).decode("utf-8")

    request_body = {
        "reference_audio": {
            "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 24000},
            "content": reference_audio_b64,
        },
        "voice_talent_consent": {
            "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 24000},
            "content": consent_audio_b64,
        },
        "consent_script": "I am the owner of this voice and I consent to Google using this voice to create a synthetic voice model.",
        "language_code": language_code,
    }

 
    headers = {
        "Authorization": f"Bearer {access_token}",
        "x-goog-user-project": project_id,
        "Content-Type": "application/json; charset=utf-8",
    }

    response = requests.post(url, headers=headers, json=request_body)
    print(response)
    response.raise_for_status()

    response_json = response.json()
    
    return response_json.get("voiceCloningKey")


def synthesize_text_with_cloned_voice(project_id, voice_key, text, audio_file_path= "output.wav", language_code="cmn-CN"):
    access_token = get_token()
    url = "https://texttospeech.googleapis.com/v1beta1/text:synthesize"

    request_body = {
        "input": {
            "text": text
        },
        "voice": {
            "language_code": language_code,
            "voice_clone": {
                "voice_cloning_key": voice_key,
            }
        },
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "sample_rate_hertz": 24000
        }
    }


    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "x-goog-user-project": project_id,
            "Content-Type": "application/json; charset=utf-8"
        }

        print(headers)
        response = requests.post(url, headers=headers, json=request_body)
        # print(response.content)
        response.raise_for_status()

        response_json = response.json()
        audio_content = response_json.get("audioContent")

        # save audio content to a file
        with open(audio_file_path, "wb") as audio_file:
            audio_file.write(base64.b64decode(audio_content))
        print(f"Audio content saved to {audio_file_path}")

        if not audio_content:
            print("Error: Audio content not found in the response.")
            print(response_json)

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
   


def read_audio_file_as_bytes(file_path):
    """
    从音频文件中读取字节数据

    参数:
    file_path (str): 音频文件路径

    返回:
    bytes: 音频文件的字节数据
    """
    try:
        with open(file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        return audio_bytes
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except Exception as e:
        print(f"读取音频文件时发生错误: {e}")
        return None


def callMiniMax(text, voice_id="jiangwen", audio_file_path="output.wav", speed=1.16, pitch=0, vol=1.33):
    """
    调用 MiniMax TTS API
    使用 env_config 中的配置而不是 streamlit session state
    """
    group_id = MINIMAX_GROUP_ID
    api_key = MINIMAX_API_KEY

    if not group_id:
        group_id = st.session_state["minimax_group_id"]

    if not api_key:
        api_key = st.session_state["minimax_api_key"]

    print(f"minimax api_key{api_key} group_id{group_id}")

    if not group_id or not api_key:
        print("请在环境变量中配置 MINIMAX_GROUP_ID 和 MINIMAX_API_KEY")
        return

    url = f"https://api.minimax.chat/v1/t2a_v2?GroupId={group_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "speech-02-hd",
        "text": text,
        "timber_weights": [
            {
                "voice_id": voice_id,
                "weight": 1
            }
        ],
        "voice_setting": {
            "voice_id": voice_id,
            "speed": speed,
            "pitch": pitch,
            "vol": vol,
            "latex_read": False
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3"
        },
        "language_boost": "auto"
    }

    response = requests.post(url, headers=headers, json=payload)

    json_data = json.loads(response.text, strict=False)
    try:
        content = json_data["data"]["audio"]
        if content:
            audio_bytes = binascii.unhexlify(content)
            # save audio content to a file
            with open(audio_file_path, "wb") as audio_file:
                audio_file.write(audio_bytes)
            print(f"Audio content saved to {audio_file_path}")
        else:
            print("Error: Audio content not found in the response.")
            print(json_data)
    except Exception as e:
        print(f"Error: {e}")
        print(json_data)


# 创建通用客户端实例
hdvoice_client = get_speech_client()


def synthesize_text_with_hd_voice(text, voice="voice", audio_file_path="output.wav", language_code="en-US", speaking_rate=1.4):
    """
    使用 HD 语音合成文本
    使用通用客户端实例
    """
    voice_name = f"{language_code}-Chirp3-HD-{voice}"
    voice = texttospeech.VoiceSelectionParams(
        name=voice_name,
        language_code=language_code,
    )

    response = hdvoice_client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=voice,
        # Select the type of audio file you want returned
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
            sample_rate_hertz=24000,
        ),
    )

    with open(audio_file_path, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio content saved to {audio_file_path}")


# 使用示例 (注释掉避免意外执行)
# synthesize_text_with_hd_voice(
#     text="Once upon a time, there was a cute cat. He was so cute that he got lots of treats.",
#     voice="Algenib",
#     audio_file_path="output.wav", 
#     language_code="en-US"
# )

# from ailib.voice_keys import voice_clone_keys
# print(voice_clone_keys["jiangwen"])
# synthesize_text_with_cloned_voice("cloud-llm-preview1", voice_clone_keys["jiangwen"], "阿尔弗雷德巧妙地问布鲁斯是对丹特的品格感兴趣，然后他深情地握住瑞秋的手，蝙蝠侠从撞击中恢复过来。他迅速跑上停车场的坡道，从高处跃下，展开披风滑翔，猛地降落在货车车顶，砸碎了挡风玻璃，制服了稻草人", audio_file_path="output.wav", language_code="cmn-CN")
# callMiniMax("你好，我是老姜", voice_id="jiangwen", audio_file_path="output.wav", speed=1.16, pitch=0, vol=1.33)
