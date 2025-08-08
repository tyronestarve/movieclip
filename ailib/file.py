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

import requests

def download_file(url, local_filename):
    """
    使用 requests 库下载文件。

    :param url: 文件的 URL 地址
    :param local_filename: 保存到本地的文件名
    """
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # 检查请求是否成功
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"文件已成功下载并保存为: {local_filename}")
        return local_filename
    except requests.exceptions.RequestException as e:
        print(f"下载文件时出错: {e}")
        return None



import magic

def get_mime_type_by_content(url):
    """
    通过文件内容识别 MIME 类型。

    :param filename: 文件名
    :return: 识别出的 MIME 类型
    """

    try:
        if "youtube" in url:
            return "video/*"
        # if "gs://" in url:

        # 發起 GET 請求
        response = requests.get(url)

        # 檢查請求是否成功 (HTTP 狀態碼 200)
        response.raise_for_status()

        # 從回應中獲取原始的 bytes 內容
        # response.content 本身就是 bytes 類型
        content_bytes = response.content

        mime = magic.Magic(mime=True)
        mime_type = mime.from_buffer(content_bytes)

        return mime_type

    except requests.exceptions.RequestException as e:
        # 處理可能發生的網絡錯誤、HTTP 錯誤等
        print(f"下载时发生错误: {e}")
        return None
    except Exception as e:
        print(f"识别 MIME 类型时出错: {e}")
        return None


# # 示例：下载一张图片
# image_url = "https://static0.xesimg.com/test-case-client-file/30f70152ccd6b.pdf"
# print(get_mime_type_by_content(image_url))
# downloaded_file = download_file(image_url, "a.a")

# if downloaded_file:
#     mime_type_content = get_mime_type_by_content(downloaded_file)
#     print(f"'{downloaded_file}' 的 MIME 类型 (通过内容识别) 是: {mime_type_content}")
