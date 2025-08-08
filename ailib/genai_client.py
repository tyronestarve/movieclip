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

import time
import asyncio
from google import genai
from google.genai import types
from google.oauth2 import service_account
from google import genai
from google.genai.types import Part
from google.cloud import storage
import cv2
import json
from ailib import file
from ailib.env_config import (
    SA_KEY_FILE_PATH, 
    GEMINI_TIMEOUT,
    GEMINI_AUTH_TYPE,
    IMAGE_AUTH_TYPE,
    VIDEO_AUTH_TYPE,
    API_KEY,
    PROXY
)
import os
import traceback


# 
# AIS : 
# 1 不支持audio_timestamp
# 2 不支持gcs
#

def _get_credentials(auth_type):
    """
    根据认证类型获取凭据
    auth_type: SA 或 ADC
    """
    credentials = None
    if auth_type == "SA":
        if SA_KEY_FILE_PATH and os.path.exists(SA_KEY_FILE_PATH):
            credentials = service_account.Credentials.from_service_account_file(
                SA_KEY_FILE_PATH, 
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
        else:
            print(f"警告: SA 认证模式但服务账户文件不存在: {SA_KEY_FILE_PATH}")
            print("回退到默认认证模式 (ADC)")

    return credentials


def init_gemini_client(project_id, location, gemini_auth_type=None):
    """
    初始化 Gemini 客户端
    根据 GEMINI_AUTH_TYPE 选择认证方式
    """

    if not gemini_auth_type:
        gemini_auth_type = GEMINI_AUTH_TYPE

    credentials = _get_credentials(gemini_auth_type)
    
    http_options = {}
    if GEMINI_TIMEOUT > 0:
        http_options["timeout"]= GEMINI_TIMEOUT
            
    if PROXY  :
        http_options["client_args"] = {'proxy': PROXY}

    if gemini_auth_type == "SA":
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            credentials=credentials,
            http_options=http_options
        )
    elif gemini_auth_type == "AIS_API_KEY":
        print("init ai studio client")
        client = genai.Client(
            api_key= API_KEY,
            http_options=http_options
        )
    elif gemini_auth_type == "ADC":
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            http_options=http_options
        )
    else:
        pass

    return client


def init_imagen_client(project_id, location):
    """
    初始化 Imagen 客户端
    根据 IMAGE_AUTH_TYPE 选择认证方式
    """
    credentials = _get_credentials(IMAGE_AUTH_TYPE)

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        credentials=credentials
    )
    
    return client


def init_veo_client(project_id, location):
    """
    初始化 Veo 客户端
    根据 VIDEO_AUTH_TYPE 选择认证方式
    """
    credentials = _get_credentials(VIDEO_AUTH_TYPE)

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        credentials=credentials
    )
    
    return client


# 保持向后兼容性的旧函数，默认使用 Gemini 客户端
def init_client(project_id, location, gemini_auth_type = None):
    """
    @deprecated 请使用 init_gemini_client, init_imagen_client, 或 init_veo_client
    """
    return init_gemini_client(project_id, location, gemini_auth_type)

async def asyncCallapi(project_id, location, model, files=[], textInputs=[], videoLinks=[], imageLinks=[], generation_config_params={},links=[], gemini_auth_type = None):
    client = init_gemini_client(project_id, location, gemini_auth_type)
    
    prompt = prepareGeminiMessages(files=files, textInputs=textInputs, videoLinks=videoLinks, imageLinks=imageLinks, links=links)
    generate_content_config = prepareGeminiConfig(generation_config_params)
    desc = ""
    try:
        start_time = time.time()
        responses =  await client.aio.models.generate_content(
                model= model,
                contents= prompt,
                config= generate_content_config,

        )
       
        end_time = time.time()
        elapsed_time = end_time - start_time
        desc =  f"""{model}
vertex ai cost={elapsed_time} second
inputtoken={responses.usage_metadata.prompt_token_count}
outputtoken={responses.usage_metadata.candidates_token_count}
thoughts_token_count={responses.usage_metadata.thoughts_token_count}
cached_content_token_count={responses.usage_metadata.cached_content_token_count}
total_token_count={responses.usage_metadata.total_token_count}
"""
        print(desc)
        return [responses,desc,model,"success"]
    except Exception as e:
        traceback.print_exc()
        return [{"error":str(e),"model":model},"error happens",model,"fail"]


def callapi(project_id, location, model, files=[], textInputs=[], videoLinks=[], imageLinks=[], generation_config_params={}, is_async = False, links=[], gemini_auth_type = None):
    client = init_gemini_client(project_id, location, gemini_auth_type)
    
    prompt = prepareGeminiMessages(files=files, textInputs=textInputs, videoLinks=videoLinks, imageLinks=imageLinks,links=links)
    generate_content_config = prepareGeminiConfig(generation_config_params)
    print(generate_content_config)
    desc = ""
    try:
        start_time = time.time()
        if is_async:
            responses =  client.aio.models.generate_content(
                model= model,
                contents= prompt,
                config= generate_content_config,

            )
        else:
            responses =  client.models.generate_content(
                model= model,
                contents= prompt,
                config= generate_content_config,
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            desc =  f"{model}  \nvertex ai cost={elapsed_time} second  \ninputtoken={responses.usage_metadata.prompt_token_count}  \noutputtoken={responses.usage_metadata.candidates_token_count}"   
            print(desc)
        return [responses,desc,model,"success"]
    except Exception as e:
        return [{"error":str(e),"model":model},"error happens",model,"fail"]

def prepareGeminiConfig(params):
    temperature = params.get("temperature", 0.01)
    response_mime_type = params.get("response_mime_type", "text/plain")
    response_modalities = params.get("response_modalities", ["TEXT"])
    systemPrompt = params.get("systemPrompt", "")
    voice_name = params.get("voice_name", None)
    audio_timestamp = params.get("audio_timestamp", False)
    grounding = params.get("grounding", False)
    code = params.get("code", False)
    include_thoughts = params.get("include_thoughts", False)
    thinking_budget = params.get("thinking_budget", None)
    max_output_tokens = params.get("max_output_tokens", 8192)
    response_schema = params.get("response_schema", None)

    generation_config = types.GenerateContentConfig(
        temperature = temperature,
        top_p = 0.95,
        max_output_tokens = max_output_tokens,
        response_modalities = response_modalities,
        safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
        )]
    )

    if audio_timestamp:
        generation_config.audio_timestamp = audio_timestamp
   
    #Parameter response_mime_type is not supported for generating image response
    if response_modalities == ["TEXT"] and response_mime_type:
        generation_config.response_mime_type = response_mime_type
    if systemPrompt:
        generation_config.system_instruction = systemPrompt

    if response_schema:
       generation_config.response_schema = response_schema
    
    if response_modalities == ["AUDIO"]:
        if not voice_name:
            voice_name = "leda"
        
        voice_config = types.VoiceConfig(
                    prebuilt_voice_config= types.PrebuiltVoiceConfig(
                        voice_name=voice_name,
                    )
                )
        generation_config = types.GenerateContentConfig(
            response_modalities = ["AUDIO"],
            speech_config = types.SpeechConfig(
                voice_config = voice_config
            )
        )

    tools = []
    if grounding:
        tools.append(types.Tool(google_search=types.GoogleSearch()))
    if code:
        tools.append(types.Tool(code_execution={}))

    if tools:
        generation_config.tools = tools

    thinking_config = {}
    if include_thoughts:
        thinking_config["include_thoughts"] = True 
    
    if  isinstance(thinking_budget, int):
        thinking_config["thinking_budget"] = thinking_budget
    
    if thinking_config:
        generation_config.thinking_config = thinking_config

    return generation_config 

  ######### gemini start ############
def prepareGeminiMessages(files=[], textInputs=[], videoLinks=[], imageLinks=[],links= []):
    prompt = []

    if files:
      for upload_file in files:
        if isinstance(upload_file, dict):
          type = upload_file["type"]
          filebytes = upload_file["value"]
        else:
          type = upload_file.type
          filebytes = upload_file.getvalue()

        fileinput = Part.from_bytes(data=filebytes, mime_type=type)
        prompt.append(fileinput)

    if imageLinks:
      for imageLink in imageLinks:
        uri = ""
        if imageLink.startswith("https://storage.googleapis.com/"):
          parts = imageLink.split("storage.googleapis.com")
          uri = "gs:/"+ parts[1]
        else:
          uri = imageLink
        
        imagePart = Part.from_uri(mime_type="image/jpeg", file_uri=uri)
        prompt.append(imagePart)

    if videoLinks:
      for videoLink in videoLinks:
        uri = ""
        if videoLink.startswith("https://storage.googleapis.com/"):
          parts = videoLink.split("storage.googleapis.com")
          uri = "gs:/"+ parts[1]
        else:
          uri = videoLink
        
        videoPart = Part.from_uri(mime_type="video/*", file_uri=uri)
        prompt.append(videoPart)
 
    if links:
       for link in links:
          mime_type = file.get_mime_type_by_content(link)
          if mime_type:
              tmppart = Part.from_uri(mime_type=mime_type, file_uri=link)
              prompt.append(tmppart)
    if textInputs:
      for textInput in textInputs:
        textInput = Part.from_text(text=textInput)
        prompt.append(textInput)

    return prompt 

# grounding search 在application/json中会返回非预期数据。
def convert_markdown_to_json(markdown_str):
    """
    将包含JSON数据的Markdown格式字符串转换为标准JSON格式

    参数:
    markdown_str (str): 包含JSON数据的Markdown格式字符串

    返回:
    dict: 标准JSON格式的字典
    """
    # 去除Markdown格式的标记

    marker = "```json"
    marker_position = markdown_str.find(marker)
    if marker_position != -1:
        # 如果找到了标记，计算标记结束后的位置
        cut_off_point = marker_position + len(marker)
        json_str = markdown_str[cut_off_point:].replace("```", "")
    elif markdown_str.startswith("```json"):
        json_str = markdown_str.strip("```json").strip("```\n")
    else:
        json_str = markdown_str
    
    # 将字符串转换为JSON格式的字典
    try:
        json_data = json.loads(json_str,strict=False)
        return json_data
    except json.JSONDecodeError as e:
        print(f"JSON解码错误: {e} ori str:{json_str}")
        return json_str

# grounding search 在application/json中会返回非预期数据。
def remove_srt_format(markdown_str):
    """
    将包含JSON数据的Markdown格式字符串转换为标准JSON格式

    参数:
    markdown_str (str): 包含JSON数据的Markdown格式字符串

    返回:
    dict: 标准JSON格式的字典
    """
    # 去除Markdown格式的标记
    
    if markdown_str.startswith("```srt"):
        outputstr = markdown_str.strip("```srt").strip("```\n")
    elif markdown_str.startswith("```"):
        outputstr = markdown_str.strip("```\n").strip("\n```") 
    else:
        outputstr = markdown_str
    
    return outputstr


async def gene_iamge(project_id = None, location = None, model = None, config = None, prompt =None):
    client = init_imagen_client(project_id, location)

    print(project_id , location, model, config, prompt)

    responses =  await client.aio.models.generate_images(model=model,config=config,prompt=prompt)
    return responses

def gene_video(project_id, location,veo_output_gcs_location, veo_params):
    client = init_veo_client(project_id, location)

    number_of_videos =  veo_params.get("veo_count",2) 
    fps =  veo_params.get("veo_fps",24)  
    duration_seconds = veo_params.get("veo_time",8)
    person_generation = "allow_adult"
    enhance_prompt =  veo_params["veo_enhance_prompt"] if "veo_enhance_prompt" in veo_params else True 
    output_gcs_uri = veo_output_gcs_location
    aspect_ratio =  veo_params.get("veo_aspect_ratio","16:9")  
    negative_prompt = veo_params.get("veo_negative_prompt","")  
    model = veo_params.get("veo_model_id","veo-2.0-generate-001") 
    prompt= veo_params.get("prompt","") 

    image= None
    if "gcs_uri" in  veo_params:
        type = "image/jpeg"
        image = types.Image(
              gcs_uri=veo_params["gcs_uri"],
              mime_type=type
          )
    elif "file" in  veo_params:
        veo_image = veo_params["file"]
        if isinstance(veo_image, dict):
          type = veo_image["type"]
          imageBytes = veo_image["value"]
        else:
            imageBytes = veo_image.getvalue()
            type = veo_image.type
        image = types.Image(
            image_bytes=imageBytes,
            mime_type=type
        )
    else:          
        pass
        
    config=types.GenerateVideosConfig(
        number_of_videos =  number_of_videos,
        fps =  fps,
        duration_seconds = duration_seconds,
        person_generation = person_generation,
        enhance_prompt =  enhance_prompt ,
        output_gcs_uri = output_gcs_uri,
        aspect_ratio = aspect_ratio, 
        negative_prompt = negative_prompt,
    )
    operation = client.models.generate_videos(
        model = model,
        prompt= prompt,
        #'A cinematic view of a dragon soaring over a medieval castle at sunset. Its wings flap powerfully as it breathes fire, with the glowing embers lighting up the sky as knights below prepare for battle.',
        config = config,
        image = image
    )
    # Poll operation
    while not operation.done:
        time.sleep(5)
        operation = client.operations.get(operation)
        print(operation)

    error = []
    veo_result = None
    if operation.result and operation.result.generated_videos:
        veo_result = operation.result
    else:
        if operation.error:
            error.append(operation.error)
        if operation.result and operation.result.rai_media_filtered_count >0 :
            error.append(operation.result.rai_media_filtered_reasons)
    return error, veo_result

###### 测试代码示例 (注释掉避免意外执行)

# project_id="oolongz-0410"
# location ="us-central1"
# veo_output_gcs_location = "gs://gemini-oolongz/veo/"
# veo_params = {
#     "prompt":"a dog"
# }
# print(callapi(project_id, location, "gemini-1.5-flash-001", files=[], textInputs=["how do you do"], videoLinks=[], imageLinks=[], generation_config_params={}, is_async = False, links=[]))
# print(gene_video(veo_project_id, location,veo_output_gcs_location, veo_params))
# 对于 async 函数，需要在 async 环境中调用:
# import asyncio
# result = asyncio.run(gene_iamge(project_id, location, "imagen-3.0-generate-001", None , "a dog"))
# print(result)


def _call_tagging_api(client, model_id, contents, generation_config):
    """Helper function to call the Gemini API for tagging."""
    return client.models.generate_content(
        model=model_id,
        contents=contents,
        config=generation_config
    )

def _merge_tag_results(results: list) -> dict:
    """Merges multiple tagging results into a single, consolidated result."""
    from collections import Counter

    all_core_genres = Counter()
    dim_counters = {
        "theme_core": Counter(), "mood_tone": Counter(), "character_archetype": Counter(),
        "setting": Counter(), "narrative_structure": Counter(), "visual_auditory_style": Counter(),
        "source_core_elements": Counter()
    }

    for res_text in results:
        try:
            data = convert_markdown_to_json(res_text)
            if not isinstance(data, dict): continue

            all_core_genres.update(data.get("core_genres", []))
            keywords = data.get("multi_dimensional_keywords", {})
            for dim, tags in keywords.items():
                if dim in dim_counters:
                    dim_counters[dim].update(tags)
        except Exception:
            continue # Ignore parsing errors for individual segments

    # Consolidate results
    final_result = {
        "core_genres": [tag for tag, count in all_core_genres.most_common(3)],
        "multi_dimensional_keywords": {}
    }
    for dim, counter in dim_counters.items():
        final_result["multi_dimensional_keywords"][dim] = [tag for tag, count in counter.most_common(2)] # Get top 2 from each dim

    return final_result


def validate_and_correct_tags(project_id, location, model_id, generated_json_str, language="English", gemini_auth_type=None):
    """
    Validates, corrects, and deduplicates generated tags in a single step.
    """
    client = init_gemini_client(project_id, location, gemini_auth_type)

    tag_map = {
        "English": {"true_story": "Based on True Story", "novel": "Based on Novel"},
        "中文": {"true_story": "基于真实故事", "novel": "基于小说"},
        "Português": {"true_story": "Baseado em Fatos Reais", "novel": "Baseado em Livro"}
    }
    
    selected_tags = tag_map.get(language, tag_map["English"])
    true_story_tag = selected_tags["true_story"]
    novel_tag = selected_tags["novel"]
    
    validation_prompt = f"""
    # ROLE
    You are a meticulous data validator, logical reasoning expert, and a specialist in data normalization and semantics.

    # TASK
    Your task is to analyze the provided JSON object, correct any logical inconsistencies, and merge any semantically duplicate tags. This must be done in a single pass.

    # RULES & INSTRUCTIONS
    1.  **Logical Correction (Highest Priority)**:
        *   A work CANNOT be tagged as both "{true_story_tag}" and "{novel_tag}" simultaneously.
        *   If you find this contradiction in the `source_core_elements` array, you MUST remove one. Prioritize "{true_story_tag}" if unsure which is more accurate.

    2.  **Semantic Deduplication**:
        *   After ensuring logical consistency, examine all tags within `core_genres` and within each array of `multi_dimensional_keywords`.
        *   Identify tags that are semantically identical but have slightly different wording (e.g., "道德模糊的角色" and "道德模糊角色"; "Anti-hero" and "Antihero").
        *   For each set of duplicates, you MUST choose only one version to keep. Prefer the most concise and standard version.
        *   This process should be case-insensitive (e.g., "action" and "Action" are the same; keep "Action").

    3.  **Output Requirements**:
        *   Return the fully cleaned, logically consistent, and deduplicated JSON object.
        *   The structure of the JSON must remain identical to the input.
        *   The language of the output JSON must be the same as the input ({language}).
        *   Your output must be ONLY the pure JSON object, with no extra text, explanations, or markdown formatting.

    # JSON for Processing
    {generated_json_str}
    """

    generation_config = types.GenerateContentConfig(
        temperature=0, top_p=0.8, max_output_tokens=8192,
        response_mime_type="application/json"
    )

    try:
        print("   - Validating, correcting, and deduplicating tags...")
        response = client.models.generate_content(
            model=model_id,
            contents=[validation_prompt],
            config=generation_config
        )
        return [response, "Validation, correction, and deduplication", model_id, "success"]
    except Exception as e:
        traceback.print_exc()
        return [{"error": str(e), "model": model_id}, "error during tag processing", model_id, "fail"]


def generate_explanations_for_tags(project_id, location, model_id, gcs_video_uri, final_tags_json_str, gemini_auth_type=None):
    """
    Generates explanations for a given set of tags based on the video content.
    """
    client = init_gemini_client(project_id, location, gemini_auth_type)
    
    explanation_prompt = f"""
    # ROLE
    You are a detail-oriented Film Analyst. Your task is to provide concrete evidence from the video for a pre-existing set of tags.

    # TASK
    For EACH tag in the provided JSON, explain WHY it was chosen by citing specific scenes, dialogues, or visual elements from the video.

    # INSTRUCTIONS
    1.  Analyze the video content thoroughly.
    2.  For every single tag provided in the `final_tags_json`, provide a brief, direct justification.
    3.  The justification must be based on observable events in the video.
    4.  Structure your output as a JSON object where keys are the tags and values are the string explanations.
    5.  Your output must be ONLY the pure JSON object, with no extra text, explanations, or markdown formatting.

    # FINAL TAGS JSON
    {final_tags_json_str}

    # OUTPUT FORMAT
    {{
      "tag_name_1": "Explanation based on a specific scene...",
      "tag_name_2": "Justification based on character actions or dialogue...",
      ...
    }}
    """

    video_part = Part.from_uri(file_uri=gcs_video_uri, mime_type="video/mp4")
    contents = [video_part, explanation_prompt]

    generation_config = types.GenerateContentConfig(
        temperature=0.2, top_p=0.8, max_output_tokens=8192,
        response_mime_type="application/json"
    )

    try:
        print("   - Generating explanations for tags...")
        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=generation_config
        )
        return [response, "Explanation generation", model_id, "success"]
    except Exception as e:
        traceback.print_exc()
        return [{"error": str(e), "model": model_id}, "error during explanation generation", model_id, "fail"]


def generate_tags_for_video(project_id, location, model_id, gcs_video_uri, prompt, response_schema,
                            enable_segmenting=False, gemini_auth_type=None):
    """
    Analyzes a video to generate tags. Supports full video and segmented video processing.
    """
    client = init_gemini_client(project_id, location, gemini_auth_type)
    generation_config = types.GenerateContentConfig(
        temperature=1, top_p=0.8, max_output_tokens=8192,
        response_schema=response_schema, response_mime_type="application/json"
    )

    try:
        if enable_segmenting:
            print("   - Mode: Segmented Video Processing")
            if isinstance(gcs_video_uri, list):
                # This part now correctly handles async execution for segmented videos
                async def run_async_tasks():
                    async_tasks = []
                    for uri in gcs_video_uri:
                        video_part = Part.from_uri(file_uri=uri, mime_type="video/mp4")
                        async_tasks.append(
                            client.aio.models.generate_content(
                                model=model_id, contents=[video_part, prompt], config=generation_config
                            )
                        )
                    return await asyncio.gather(*async_tasks)
                
                responses = asyncio.run(run_async_tasks())
                
                final_text = json.dumps(_merge_tag_results([r.text for r in responses]))
                
                # Manually construct the response object as from_dict is not available.
                part = types.Part(text=final_text)
                content = types.Content(parts=[part])
                candidate = types.Candidate(content=content, finish_reason="STOP", safety_ratings=[])
                mock_response = types.GenerateContentResponse(candidates=[candidate])
                
                return [mock_response, "Merged result from segments", model_id, "success"]
            else:
                raise ValueError("Segmented processing requires a list of GCS URIs.")

        print("   - Mode: Full Video Processing")
        video_part = Part.from_uri(file_uri=gcs_video_uri, mime_type="video/mp4")
        contents = [video_part, prompt]

        print(f"   - Calling Gemini API ({model_id})...")
        response = _call_tagging_api(client, model_id, contents, generation_config)
        return [response, "Single video analysis", model_id, "success"]

    except Exception as e:
        traceback.print_exc()
        return [{"error": str(e), "model": model_id}, "error happens", model_id, "fail"]
