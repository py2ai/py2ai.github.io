---
description: Step-by-step beginner-friendly guide to create a voice-enabled PyGame wall clock with tick sound, date display, TTS time announcement, and STT voice recognition using Vosk.
featured-img: 20251114-wallclock-greeting
keywords:
- Python
- PyGame
- clock
- wall-clock
- tick sound
- real-time
- TTS
- text-to-speech
- STT
- speech-to-text
- Vosk
- beginner
- tutorial
layout: post
mathjax: false
tags:
- python
- pygame
- clock
- sound
- tts
- stt
- text-to-speech
- speech-to-text
- vosk
- beginner
- tutorial
title: "听和说的时钟"
lang: zh
en_url: /Listening-and-Talking-clock/
ko_url: /ko/Listening-and-Talking-clock/
---
# 带语音时间和问候语的挂钟

本教程展示了如何使用创建 **基于 Python 的挂钟** `pygame`，使用 **pyttsx3** 和 **Vosk** 进行 **文本转语音** 和 **语音转文本**。该应用程序侦听“时间”一词，并根据当前时间响应当前时间和问候语。

<div class="video-container">
  <iframe 
    width="560" 
    height="315" 
    src="https://www.youtube.com/embed/bjY2wvX-3B8" 
    title="YouTube video player" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen>
  </iframe>
</div>

## 介绍

该项目使用以下命令构建了一个**漂亮的挂钟 GUI** `pygame`——但有一个转折：

* 它可以**大声说出时间** ...并且它可以使用语音识别**听到您询问时间**。
* 当您说**“时间”**时，应用程序将使用**Vosk**检测您的语音，使用**pyttsx3**说出当前时间，并在屏幕底部显示流畅的**打字动画**。

## 特点概述

### 模拟挂钟

- 光滑的秒针、分针和时针
- 日期和星期显示
- 可选的深色主题兼容

### 内置滴答声

- 使用 NumPy 人工生成
- 无需外部音频文件

### 语音检测（STT）

- 使用 Vosk 离线语音识别
- 无需互联网即可工作
- 检测简单的关键字（“时间”）

### 文本转语音 (TTS)

- 使用 pyttsx3（离线）
- 自动说话：
  *“下午好。现在是下午 03:25！”*

### 打字动画

- 显示问候语和时间
- 平滑闪烁的光标
- 几秒钟后自动清除

### 收听按钮

- 切换连续麦克风监听
- 在后台线程中运行识别

## 先决条件

您只需要：

- Python 3.8+（最好使用Python 3.12）
- 麦克风
- 终端基本使用
- 能够安装软件包

## 安装依赖项

### 视窗

```bash
python -m venv py312
py312\Scripts\activate

pip install pygame pyttsx3 sounddevice vosk numpy
```

下载 Vosk 模型：
https://alphacephei.com/vosk/models

获取模型例如[this](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip“https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip”)

提取并重命名：

```
vosk-model-small-en-us-0.15
```

### macOS

```bash
python3 -m venv py312
source py312/bin/activate
brew install portaudio
pip install pygame pyttsx3 sounddevice vosk numpy
```

下载与上面相同的英文模型。
--------------------------

### Linux

```bash
python3 -m venv py312
source py312/bin/activate

pip install pygame pyttsx3 sounddevice vosk numpy
sudo apt update
sudo apt install -y libportaudio2 libportaudiocpp0 portaudio19-dev
```

## 了解语音转文本 (Vosk)

语音转文本 (STT) 是将口语转换为书面文本的过程。 Vosk 是最流行的离线 STT 引擎之一，以轻量、准确且易于在 Python 项目中使用而闻名。

以下是适合教程、文档或学习目的的详细说明。

### 为什么语音转文本很重要

语音转文本技术在现代软件中已变得至关重要，因为：

#### 免提交互

用户可以使用语音控制应用程序，这对于时钟、助手和任何需要手动操作的场景（烹饪、驾驶等）很有用。

#### 无障碍

STT 可以帮助有运动障碍或无法轻松打字的用户。

#### 实时自动化

语音命令可以立即触发事件 - 例如，
“启动计时器”、“停止音乐”、“现在几点了”。

#### 无需屏幕即可工作

适用于 IoT 设备、Raspberry Pi 系统或嵌入式小工具。

#### 离线安全

Vosk 完全离线工作，因此不会将语音数据发送到云端，从而增强了隐私性。

### Vosk 的工作原理——理论（简化）

尽管 Vosk 感觉使用起来很简单，但它实际上使用了严格的语音处理理论。这是一个易于理解、适合初学者的解释：

1. 音频采集

* 您的麦克风记录原始音频波。
* 这些波只是代表气压随时间变化的数字。

2. 特征提取（MFCC）

* 对于机器学习模型来说，原始音频过于详细且嘈杂。
* Vosk 将原始音频转换为 MFCC 特征（梅尔倒谱系数）。

#### MFCC 代表：

- 频率分布
- 响度
- 语气
- 人类感知为言语的模式

*将 MFCC 视为神经网络可以理解的声音指纹。*

3.声学模型（神经网络）
该模型采用 MFCC 特征并预测音素 —
最小的声音单位，例如：
`k    a    t    ( = "cat" )`
声学模型经过数千小时的语音录音训练。
4. 语言模型
人类不会以随机的音素序列说话。
因此，语言模型有助于预测哪些单词有意义。

例如：
如果声学模型检测到以下内容：
`d   t   a   m   p`
语言模型引导它：
`→ "time"`
而不是胡言乱语。
5. 解码器
解码器结合了：

- 声学模型的预测
- 来自语言模型的概率
  and chooses the most likely final text output.
  Result: clear, readable text.

### 为什么开发人员喜欢 Vosk

* 100% 离线
* 没有互联网意味着：
  ✔ 隐私
  ✔ 可靠性
  ✔ 非常适合物联网或现场环境
* 低CPU使用率

运行于：

- 树莓派
- 旧笔记本电脑
- 中档电脑
- 提供小型型号
- 某些型号<50MB。
- 快速且实时
- 即使在普通的硬件上，它也可以立即转录。
- 多语言支持

### 蜡模型类型

您可以根据您的设备进行选择：

#### 小型号

- <40MB
- 最快
- 精度较低
- 非常适合 Raspberry Pi 或简单命令
- 非常适合这个“语音时钟项目”

#### 中型型号

- 平衡精度+速度
- 适用于台式机或笔记本电脑

#### 大型机型

- 最佳准确度
- CPU负载较重
- 对于简单的语音命令来说太过分了

### 从哪里获取语言模型

所有官方型号在这里：
https://alphacephei.com/vosk/models

### 支持的语言

沃斯克支持：

| 语言     | 型号                            |
| -------- | ------------------------------- |
| 英语     | `vosk-model-small-en-us-0.15` |
| 日语     | `vosk-model-small-ja-0.22`    |
| 中文     | `vosk-model-small-cn-0.22`    |
| 西班牙语 | `vosk-model-small-es-0.42`    |
| 法语     | `vosk-model-small-fr-0.22`    |
| 印地语   | `vosk-model-small-hi-0.22`    |

……还有更多。

### 初学者应该使用哪种模型？

使用**小模型**：

- 快速地
- 低CPU使用率
- 非常适合树莓派
- 对于单字命令来说足够准确

小型号名称示例：

`vosk-model-small-en-us-0.15`
`vosk-model-small-es-0.42`
`vosk-model-small-fr-0.22`

## 了解文本转语音 (pyttsx3)

### 改变声音

在代码中：

```python
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
```

### 改变说话速度

```python
engine.setProperty('rate', 150)
```

共同价值观：

- 120（慢速）
- 150（默认）
- 180（快速）

## 代码分解

### 时钟渲染

时钟是手动绘制的：

- 外圈
- 小时数
- 分钟刻度
- 根据时间旋转指针

### 滴答声生成

而不是加载 `.wav`，我们生成音频：

- 1500赫兹点击
- 50 毫秒持续时间
- 指数褪色

感谢 NumPy，时钟始终滴答作响，无需导入外部文件。

### 打字动画

问候语看起来就像真实的打字一样：

- 人物逐渐出现
- 光标闪烁
- 4秒后，文字自动清除

### 监听按钮行为

- 打开/关闭
- 蓝色 → 闲置
- 绿色 → 聆听
- 在后台运行 Vosk 麦克风流

### STT回调逻辑

当 Vosk 解码语音时：

- 打印检测到的文本
- 如果包含“时间”，则调用 `speak_time()`

## 运行应用程序

一切安装完毕后：

```bash
python main.py
```

步骤：

1.时钟出现
2. 单击**听**
3. 说：**“时间”**
4. 时钟会说出当前时间
5.底部出现文字动画

## 故障排除

### ❗ 未检测到麦克风

尝试：

```bash
pip install sounddevice
```

或者选择输入设备：

```python
sd.default.device = 1
```

### ❗ 未检测到语音

使用**小**模型；大的需要更多的CPU。
发音清晰，点击“听”后等待 1-2 秒。

### ❗ TTS 只能运行一次

确保每个 TTS 调用都会创建一个**新引擎**（已在提供的代码中完成）。

## 完整源代码

### 1.Windows DPI 感知

```python
import ctypes
try:
    ctypes.windll.user32.SetProcessDPIAware()
except:
    pass
```

- 确保应用程序在 Windows 中的**高 DPI 屏幕**上正确显示。
- 包裹在 `try`块以与其他操作系统兼容。

### 2. 进口

```python
import pygame, math, datetime, sys, numpy as np, pyttsx3, threading, time
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json, os
```

- **pygame**：GUI 和图形。
- **数学**：时钟指针的三角学。
- **日期时间**：时钟和问候语的当前时间。
- **numpy**：生成人工滴答声。
- **pyttsx3**：文本转语音引擎。
- **线程**：在后台运行 TTS/STT。
- **声音设备和vosk**：语音转文本识别。
- **json & os**：解析 Vosk 输出并处理文件。

### 3.Pygame初始化

```python
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PyShine Wall Clock")
```

- 初始化 **Pygame** 和 **音频混合器** 以进行声音播放。
- 设置**屏幕尺寸**和窗口**标题**。

### 4. 常数和颜色

```python
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRAY = (150, 150, 150)
DARK_GRAY = (50, 50, 50)
BUTTON_COLOR = (0, 128, 255)
BUTTON_HOVER = (0, 180, 255)
BUTTON_ACTIVE = (0, 200, 0)
LIME = (0, 255, 0)
```

- 定义用于钟面、指针、按钮和文本的**颜色**。

### 5. 时钟参数及字体

```python
center_x, center_y = WIDTH // 2, HEIGHT // 2
clock_radius = 150
font = pygame.font.SysFont('Arial', 24, bold=True)
date_font = pygame.font.SysFont('Arial', 20)
button_font = pygame.font.SysFont('Arial', 20, bold=True)
time_str_font = pygame.font.SysFont('Arial', 28, bold=True)
```

`center_x, center_y`：时钟中心。
`clock_radius`：钟面的尺寸。

- **数字、日期、按钮文本和 TTS 文本显示**的字体。

### 6. 滴答声

```python
def create_tick_sound():
    ...
    tick_sound = pygame.sndarray.make_sound(sound_array)
    tick_sound.set_volume(0.5)
    return tick_sound
```

- 使用 NumPy 生成 **短 1500Hz 的点击**。
- 无需外部音频文件。
- 用于播放**每秒滴答声**。

### 7. 收听按钮

```python
button_rect = pygame.Rect(WIDTH // 2 - 80, 80, 160, 50)
listening_active = False
def draw_button(mouse_pos):
    ...
```

- 在屏幕上绘制**按钮**。
  Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.
- 控制**麦克风监听状态**。

### 8. 文本输入和 TTS

```python
def speak_time():
    ...
    threading.Thread(target=tts_func, args=(spoken_time_str,), daemon=True).start()
```

- 根据当前时间确定**问候语**。
- 格式**语音文本**：例如，`"Good afternoon\nIt's 03:25 PM now!"`。
- 在后台线程中启动**文本转语音**。
- 更新**打字动画**变量。

### 9.Wax 语音转文本设置

```python
MODEL_PATH = "vosk-model-small-en-us-0.15"
vosk_model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, 16000)
```

- 加载**离线 Vosk 模型**。
- 识别器将**音频字节转换为文本**。
- 确保**离线语音识别**。

#### STT 回调

```python
def stt_callback(indata, frames, time_data, status):
    ...
    if "time" in result_text.lower():
        speak_time()
```

- 处理来自麦克风的音频。
- 将其转换为文本。
- 触发器 `speak_time()`当检测到**关键字“时间”**时。

### 10. 时钟绘图函数

#### 钟面

```python
def draw_clock_face():
    ...
```

- 绘制**外圈、小时数字、分钟刻度**。
- 区分**小时刻度**（较粗）和**分钟刻度**（较细）。

#### 钟针

```python
def draw_clock_hands():
    ...
```

- 根据当前时间绘制**时针、分针、秒针**。
- 每秒播放**滴答声**。
- 绘制**中心枢轴**圆。

#### 日期显示

```python
def draw_date_display(now):
    ...
```

- 显示**当前日期**和**星期几**。

#### 打字动画

```python
def draw_spoken_time():
    ...
```

- 像打字一样逐渐显示**问候语和时间**。
- 光标**闪烁**。
- **4 秒**后自动清除。

### 11. 主循环

```python
def main():
    ...
```

- 处理**事件**：
- 辞职
- ESC键
- 鼠标点击**收听按钮**
- 更新：
- **钟面**
- **手**
- **日期**
- **输入问候语**
- **收听按钮**
- 以 **30 FPS** 运行。
- 确保**流畅的动画和交互**。

### 12. 入口点

```python
if __name__ == "__main__":
    main()
```

- 直接执行脚本时启动**主循环**。

### 主要.py

完整的工作源代码在这里：
{% include codeHeader.html %}

```python
# Tutorial and Source Code available: www.pyshine.com

import ctypes
try:
    ctypes.windll.user32.SetProcessDPIAware()
except:
    pass
import pygame
import math
import datetime
import sys
import numpy as np
import pyttsx3
import threading
import time

#  VOSK STT IMPORTS 
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json
import os

# Initialize Pygame
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)

# Screen dimensions
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PyShine Wall Clock")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRAY = (150, 150, 150)
DARK_GRAY = (50, 50, 50)
BUTTON_COLOR = (0, 128, 255)
BUTTON_HOVER = (0, 180, 255)
BUTTON_ACTIVE = (0, 200, 0)
LIME = (0, 255, 0)

# Clock parameters
center_x, center_y = WIDTH // 2, HEIGHT // 2
clock_radius = 150

# Fonts
font = pygame.font.SysFont('Arial', 24, bold=True)
date_font = pygame.font.SysFont('Arial', 20)
button_font = pygame.font.SysFont('Arial', 20, bold=True)
time_str_font = pygame.font.SysFont('Arial', 28, bold=True)

# Tick sound
def create_tick_sound():
    sample_rate = 44100
    duration = 0.05
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    envelope = np.exp(-50 * t)
    waveform = 0.5 * envelope * np.sign(np.sin(2 * np.pi * 1500 * t))
    waveform_int16 = np.int16(waveform * 3276)
    sound_array = np.column_stack([waveform_int16, waveform_int16])
    tick_sound = pygame.sndarray.make_sound(sound_array)
    tick_sound.set_volume(0.5)
    return tick_sound

tick = create_tick_sound()
last_second = -1

# Button
button_rect = pygame.Rect(WIDTH // 2 - 80,  80, 160, 50)
listening_active = False  # Button state
printed=False
def draw_button(mouse_pos):
    global printed
    if listening_active:
        color = BUTTON_ACTIVE
        text_str = "LISTENING..."
        if  printed==False:
            print('Start listening...')
            printed=True
    else:
        color = BUTTON_HOVER if button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
        text_str = "LISTEN"
        printed=False
    pygame.draw.rect(screen, color, button_rect, border_radius=10)
    text = button_font.render(text_str, True, WHITE)
    text_rect = text.get_rect(center=button_rect.center)
    screen.blit(text, text_rect)

# Shared variables
spoken_time_str = ""
typed_text = ""
typing_start_time = 0
typing_speed = 8
cursor_visible = True
last_cursor_toggle = 0
text_display_complete_time = None

# Speak time and trigger typing
def speak_time():
    global spoken_time_str, typed_text, typing_start_time, text_display_complete_time
    now = datetime.datetime.now()
    hour, minute = now.hour, now.minute

    # Determine AM/PM
    am_pm = "AM" if hour < 12 else "PM"
    hour_display = hour % 12
    hour_display = 12 if hour_display == 0 else hour_display

    # Determine greeting based on hour
    if 5 <= hour < 12:
        greeting = "Good morning"
    elif 12 <= hour < 17:
        greeting = "Good afternoon"
    elif 17 <= hour < 21:
        greeting = "Good evening"
    else:
        greeting = "Good night"

    # Combine greeting and time as two lines
    spoken_time_str = f"{greeting}\nIt's {hour_display:02d}:{minute:02d} {am_pm} now!"

    typed_text = ""  # Reset typing
    typing_start_time = time.time()
    text_display_complete_time = None

    # Speak TTS
    def tts_func(text):
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_engine.say(text.replace("\n", ". "))  # Speak as single sentence
        tts_engine.runAndWait()

    threading.Thread(target=tts_func, args=(spoken_time_str,), daemon=True).start()

#  VOSK STT SETUP
MODEL_PATH = "vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    print(f"Missing model folder '{MODEL_PATH}'")
    sys.exit(1)

vosk_model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, 16000)
sd_stream = None  # Global reference to microphone stream

def audio_to_bytes(indata):
    try:
        return bytes(indata)
    except:
        return indata.tobytes()

def stt_listen_loop():
    global sd_stream
    try:
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype='int16',
            channels=1,
            callback=stt_callback
        ) as stream:
            sd_stream = stream
            while listening_active:
                time.sleep(0.1)
    except Exception as e:
        print("Microphone error:", e)

def stt_callback(indata, frames, time_data, status):
    if status:
        print("Audio status:", status)
    data = audio_to_bytes(indata)
    if recognizer.AcceptWaveform(data):
        result_text = json.loads(recognizer.Result()).get("text", "")
        if result_text.strip():  # Only print non-empty text
            print(f"Detected: {result_text}")
        if "time" in result_text.lower():
            speak_time()

# CLOCK DRAWING FUNCTIONS

def draw_clock_face():
    pygame.draw.circle(screen, WHITE, (center_x, center_y), clock_radius, 2)
    pygame.draw.circle(screen, DARK_GRAY, (center_x, center_y), clock_radius - 5, 2)
    for hour in range(1, 13):
        angle = math.radians(hour * 30 - 90)
        number_x = center_x + (clock_radius - 30) * math.cos(angle) - 10
        number_y = center_y + (clock_radius - 30) * math.sin(angle) - 10
        number_text = font.render(str(hour), True, WHITE)
        screen.blit(number_text, (number_x, number_y))
        tick_start_x = center_x + (clock_radius - 15) * math.cos(angle)
        tick_start_y = center_y + (clock_radius - 15) * math.sin(angle)
        tick_end_x = center_x + (clock_radius - 5) * math.cos(angle)
        tick_end_y = center_y + (clock_radius - 5) * math.sin(angle)
        pygame.draw.line(screen, WHITE, (tick_start_x, tick_start_y), (tick_end_x, tick_end_y), 3)
    for minute in range(60):
        if minute % 5 != 0:
            angle = math.radians(minute * 6 - 90)
            tick_start_x = center_x + (clock_radius - 10) * math.cos(angle)
            tick_start_y = center_y + (clock_radius - 10) * math.sin(angle)
            tick_end_x = center_x + (clock_radius - 5) * math.cos(angle)
            tick_end_y = center_y + (clock_radius - 5) * math.sin(angle)
            pygame.draw.line(screen, GRAY, (tick_start_x, tick_start_y), (tick_end_x, tick_end_y), 1)

def draw_clock_hands():
    global last_second
    now = datetime.datetime.now()
    hour, minute, second = now.hour % 12, now.minute, now.second
    if second != last_second:
        tick.play()
        last_second = second
    hour_angle = math.radians(hour * 30 + minute * 0.5 - 90)
    minute_angle = math.radians(minute * 6 + second * 0.1 - 90)
    second_angle = math.radians(second * 6 - 90)
    hour_x = center_x + clock_radius * 0.5 * math.cos(hour_angle)
    hour_y = center_y + clock_radius * 0.5 * math.sin(hour_angle)
    pygame.draw.line(screen, WHITE, (center_x, center_y), (hour_x, hour_y), 6)
    minute_x = center_x + clock_radius * 0.7 * math.cos(minute_angle)
    minute_y = center_y + clock_radius * 0.7 * math.sin(minute_angle)
    pygame.draw.line(screen, WHITE, (center_x, center_y), (minute_x, minute_y), 4)
    second_x = center_x + clock_radius * 0.8 * math.cos(second_angle)
    second_y = center_y + clock_radius * 0.8 * math.sin(second_angle)
    pygame.draw.line(screen, RED, (center_x, center_y), (second_x, second_y), 2)
    pygame.draw.circle(screen, RED, (center_x, center_y), 8)
    pygame.draw.circle(screen, WHITE, (center_x, center_y), 8, 2)
    return now

def draw_date_display(now):
    date_text = date_font.render(now.strftime("%Y-%m-%d"), True, WHITE)
    day_text = date_font.render(now.strftime("%A").upper(), True, WHITE)
    date_rect = date_text.get_rect(midtop=(center_x, center_y - clock_radius + 70))
    day_rect = day_text.get_rect(midtop=date_rect.midbottom)
    screen.blit(date_text, date_rect)
    screen.blit(day_text, day_rect)

def draw_spoken_time():
    global typed_text, last_cursor_toggle, cursor_visible, text_display_complete_time, spoken_time_str
    if spoken_time_str:
        elapsed = time.time() - typing_start_time
        # Split into lines
        lines = spoken_time_str.split("\n")
        chars_to_show = min(int(elapsed * typing_speed), sum(len(line) for line in lines))
  
        # Determine how many chars to show per line
        display_lines = []
        chars_remaining = chars_to_show
        for line in lines:
            if chars_remaining > len(line):
                display_lines.append(line)
                chars_remaining -= len(line)
            else:
                display_lines.append(line[:chars_remaining])
                break
  
        # Clear after 4 seconds of full display
        if chars_to_show == sum(len(line) for line in lines) and text_display_complete_time is None:
            text_display_complete_time = time.time()
        if text_display_complete_time and (time.time() - text_display_complete_time > 4):
            spoken_time_str = ""
            typed_text = ""
            return

        # Cursor blink timer
        if time.time() - last_cursor_toggle > 0.5:
            cursor_visible = not cursor_visible
            last_cursor_toggle = time.time()

        # Render each line
        y_offset = HEIGHT - 130
        for i, line in enumerate(display_lines):
            text_surface = time_str_font.render(line, True, LIME)
            text_rect = text_surface.get_rect(center=(WIDTH // 2, y_offset + i*35))
            screen.blit(text_surface, text_rect)

        # Draw cursor at end of last line
        if cursor_visible and display_lines:
            last_line = display_lines[-1]
            text_surface = time_str_font.render(last_line, True, LIME)
            text_rect = text_surface.get_rect(center=(WIDTH // 2, y_offset + (len(display_lines)-1)*35))
            cursor_x = text_rect.right + 2
            cursor_y = text_rect.top + 4
            cursor_height = text_rect.height - 2
            pygame.draw.rect(screen, LIME, (cursor_x, cursor_y-4, 3, cursor_height))


# MAIN LOOP
def main():
    global listening_active
    clock = pygame.time.Clock()
    running = True
    stt_thread = None

    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if button_rect.collidepoint(event.pos):
                    listening_active = not listening_active
                    if listening_active:
                        # Start background STT listening
                        stt_thread = threading.Thread(target=stt_listen_loop, daemon=True)
                        stt_thread.start()
                    else:
                        # Stop listening
                        print("Stopping listening...")
                        sd_stream = None

        screen.fill(BLACK)
        draw_clock_face()
        now = draw_clock_hands()
        draw_date_display(now)
        draw_spoken_time()
        draw_button(mouse_pos)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

```

**网站：** https://www.pyshine.com
**作者：** PyShine
