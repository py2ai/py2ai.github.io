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
title: 인사말이 포함된 음성 인식 벽시계
lang: ko
en_url: /Listening-and-Talking-clock/
zh_url: /zh/Listening-and-Talking-clock/
ja_url: /ja/Listening-and-Talking-clock/
---

# 음성 시간과 인사말이 포함된 벽시계

이 튜토리얼에서는 다음을 사용하여 **Python 기반 벽시계**를 만드는 방법을 보여줍니다.`pygame`, **pyttsx3** 및 **Vosk**를 사용하여 **text-to-speech** 및 **speech-to-text**를 사용합니다. 앱은 "시간"이라는 단어를 듣고 현재 시간과 현재 시간을 기준으로 한 인사말로 응답합니다.

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

---

# 목차

1. [소개](#introduction)
2. [기능 개요](#features-overview)
3. [전제 조건](#prerequisites)
4. [종속성 설치](#installing-dependencies)
    - [윈도우](#windows)
    - [macOS](#macos)
    - [리눅스](#linux)
5. [음성-텍스트 이해(Vosk)](#understanding-speech-to-text-vosk)
    - [음성-텍스트 변환이 중요한 이유](#why-speech-to-text-is-important)
    - [Vosk의 작동 원리 - 이론(간체)](#how-vosk-works--the-theory-simplified)
    - [왁스 모델 유형](#wax-model-types)
    - [언어 모델을 얻을 수 있는 곳](#where-to-get-언어-모델)
    - [지원 언어](#supported-언어)
    - [초보자는 어떤 모델을 사용해야 할까요?](#which-model-should-beginners-use)
6. [텍스트 음성 변환(pyttsx3) 이해](#understanding-text-to-speech-pyttsx3)
    - [목소리 바꾸기](#changing-voice)
    - [말하는 속도 변경](#changing-speaking-speed)
7. [코드 분석](#code-breakdown)
    - [시계 렌더링](#clock-rendering)
    - [틱소리 생성](#틱-소리-생성)
    - [타자 애니메이션](#typing-animation)
    - [듣기 버튼 동작](#listen-button-behavior)
    - [STT 콜백 로직](#stt-callback-logic)
8. [앱 실행](#running-the-app)
9. [문제 해결](#troubleshooting)
10. [전체 소스 코드](#full-source-code)

## 소개

이 프로젝트는 다음을 사용하여 **아름다운 벽시계 GUI**를 구축합니다.`pygame`— 하지만 반전이 있습니다.

* **시간을 소리내어 말**할 수 있으며 음성 인식을 통해 **시간을 묻는 소리**를 들을 수 있습니다.
* **“시간”**이라고 말하면 앱은 **Vosk**를 사용하여 음성을 감지하고 **pyttsx3**을 사용하여 현재 시간을 말하고 화면 하단에 부드러운 **타이핑 애니메이션**을 표시합니다.

## 기능 개요

### 아날로그 벽시계

- 부드러운 초침, 분침, 시침
- 날짜 및 요일 표시
- 선택적으로 어두운 테마와 호환 가능

### 내장 틱 사운드

- NumPy를 사용하여 인위적으로 생성됨
- 외부 오디오 파일이 필요하지 않습니다.

### 음성 감지(STT)

- Vosk 오프라인 음성 인식 사용
- 인터넷 없이 작동
- 단순 키워드(“시간”)를 탐지합니다.

### TTS(텍스트 음성 변환)

- pyttsx3 사용(오프라인)
- 자동으로 말합니다:
*"안녕하세요. 지금은 오후 3시 25분입니다!"*

### 타이핑 애니메이션

- 인사말과 시간을 표시합니다.
- 부드럽게 깜박이는 커서
- 몇 초 후에 자동으로 지워집니다.

### 듣기 버튼

- 지속적인 마이크 청취를 전환합니다.
- 백그라운드 스레드에서 인식을 실행합니다.

## 전제 조건

다음만 필요합니다:

- Python 3.8+(Python 3.12를 사용하는 것이 더 좋음)
- 마이크
- 기본 단말기 사용법
- 패키지 설치 기능


## 종속성 설치

### 윈도우

```bash
python -m venv py312
py312\Scripts\activate

pip install pygame pyttsx3 sounddevice vosk numpy
```

Vosk 모델 다운로드:
https://alphacephei.com/vosk/models

예를 들어 모델을 얻으세요. [이것](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")

추출 및 이름 바꾸기:

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

위와 동일한 영어 모델을 다운로드합니다.
----------------------------

### 리눅스

```bash
python3 -m venv py312
source py312/bin/activate

pip install pygame pyttsx3 sounddevice vosk numpy
sudo apt update
sudo apt install -y libportaudio2 libportaudiocpp0 portaudio19-dev
```



## 음성-텍스트 이해(Vosk)
STT(Speech-to-Text)는 음성 언어를 서면 텍스트로 변환하는 프로세스입니다. Vosk는 Python 프로젝트에서 가볍고 정확하며 사용하기 쉬운 것으로 알려진 가장 인기 있는 오프라인 STT 엔진 중 하나입니다.

다음은 튜토리얼, 문서화 또는 학습 목적에 적합한 자세한 설명입니다.

### 음성-텍스트 변환이 중요한 이유
Speech-to-Text 기술은 다음과 같은 이유로 현대 소프트웨어에서 필수적이 되었습니다.
#### 핸즈프리 상호작용
사용자는 음성을 사용하여 앱을 제어할 수 있으며 시계, 보조자 및 손이 바쁜 상황(요리, 운전 등)에 유용합니다.
#### 접근성
STT는 운동 장애가 있거나 쉽게 입력할 수 없는 사용자를 돕습니다.
#### 실시간 자동화
음성 명령은 즉시 이벤트를 트리거할 수 있습니다. 예:
"타이머 시작", "음악 중지", "지금 몇 시야".
#### 화면 없이 작동
IoT 장치, Raspberry Pi 시스템 또는 임베디드 장치에 유용합니다.
#### 오프라인 보안
Vosk는 완전히 오프라인으로 작동하므로 음성 데이터가 클라우드로 전송되지 않아 개인 정보 보호가 강화됩니다.

### Vosk 작동 방식 - 이론(간체)
Vosk는 사용이 간편하다고 느껴지지만 내부적으로는 심각한 음성 처리 이론을 사용합니다. 다음은 이해하기 쉽고 초보자에게 친숙한 설명입니다.

1. 오디오 캡처

* 마이크는 원시 오디오 웨이브를 녹음합니다.
* 이 파동은 시간에 따른 기압 변화를 나타내는 숫자일 뿐입니다.
2. 특징 추출(MFCC)

* 원시 오디오는 기계 학습 모델에 비해 너무 자세하고 잡음이 많습니다.
* Vosk는 원시 오디오를 MFCC 기능(Mel-Frequency Cepstral Coefficients)으로 변환합니다.

#### MFCC는 다음을 나타냅니다.


- 주파수 분포
- 음량
- 톤
- 인간이 말로 인식하는 패턴

*MFCC를 신경망이 이해할 수 있는 소리의 지문이라고 생각하세요.*

3. 음향 모델(신경망)
이 모델은 MFCC 기능을 사용하여 음소를 예측합니다.
다음과 같은 소리의 가장 작은 단위:
`k    a    t    ( = "cat" )`
음향 모델은 수천 시간의 음성 녹음을 통해 훈련되었습니다.
4. 언어 모델
인간은 임의의 음소 순서로 말하지 않습니다.
따라서 언어 모델은 어떤 단어가 의미가 있는지 예측하는 데 도움이 됩니다.

예를 들어:
음향 모델이 다음과 같은 것을 감지하는 경우:
`d   t   a   m   p`
언어 모델은 다음을 안내합니다.
`→ "time"`
횡설수설 대신.
5. 디코더
디코더는 다음을 결합합니다.

- 음향 모델로부터의 예측
- 언어 모델의 확률
     and chooses the most likely final text output.
     Result: clear, readable text.

### 개발자들이 Vosk를 좋아하는 이유

* 100% 오프라인
* 인터넷이 없다는 것은 다음을 의미합니다.
✔ 개인정보 보호
✔ 신뢰성
✔ IoT 또는 현장 환경에 적합
* 낮은 CPU 사용량

실행 대상:

- 라즈베리 파이
- 오래된 노트북
- 중급 PC
- 소형 모델 이용 가능
- 일부 모델은 50MB 미만입니다.
- 빠르고 실시간
- 보통 수준의 하드웨어에서도 즉시 기록됩니다.
- 다국어 지원

### 왁스 모델 유형

귀하의 장치에 따라 선택할 수 있습니다:

#### 소형 모델

- <40MB
- 가장 빠름
- 낮은 정확도
- Raspberry Pi 또는 간단한 명령에 이상적입니다.
- 이 "음성 시계 프로젝트"에 딱 맞습니다.

#### 중형 모델

- 균형 잡힌 정확도 + 속도
- 데스크탑이나 노트북에 적합

#### 대형 모델

- 최고의 정확도
- CPU 부하가 더 커짐
- 간단한 음성 명령에는 과잉

### 언어 모델을 얻을 수 있는 곳

모든 공식 모델은 여기에 있습니다:
https://alphacephei.com/vosk/models

### 지원되는 언어

Vosk는 다음을 지원합니다.

| 언어 | 모델 |
| -------- | ------------------ |
| 영어 |`vosk-model-small-en-us-0.15`
| 일본어 |`vosk-model-small-ja-0.22`
| 중국어 |`vosk-model-small-cn-0.22`
| 스페인어 |`vosk-model-small-es-0.42`
| 프랑스어 |`vosk-model-small-fr-0.22`
| 힌디어 |`vosk-model-small-hi-0.22`

…그리고 더 많은 것들이 있습니다.

### 초보자는 어떤 모델을 사용해야 합니까?

**작은 모델** 사용:

- 빠른
- 낮은 CPU 사용량
- 라즈베리 파이에 딱 맞습니다.
- 한 단어 명령에도 충분히 정확함

작은 모델 이름의 예:

`vosk-model-small-en-us-0.15`
`vosk-model-small-es-0.42`
`vosk-model-small-fr-0.22`



## 텍스트 음성 변환(pyttsx3) 이해

### 목소리 바꾸기

코드에서:

```python
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
```

### 말하기 속도 변경

```python
engine.setProperty('rate', 150)
```

공통 값:

- 120 (느림)
- 150(기본값)
- 180 (빠름)



## 코드 분석

### 시계 렌더링

시계는 수동으로 그려집니다.

- 외부 원
- 시간 번호
- 분 틱
- 시간에 따라 바늘을 회전

### 틱 사운드 생성

로딩하는 대신`.wav`, 오디오를 생성합니다.

- 1500Hz 클릭
- 지속 시간 50ms
- 지수 페이드

NumPy 덕분에 외부 파일을 가져오지 않고도 시계가 항상 똑딱거립니다.

### 타이핑 애니메이션

인사말은 실제 입력하는 것처럼 나타납니다.

- 캐릭터가 점차적으로 나타남
- 커서가 깜박입니다.
- 4초 후 텍스트가 자동으로 지워집니다.

### 듣기 버튼 동작

- 켜기/끄기 전환
- 파란색 → 유휴 상태
- 녹색 → 듣기
- 백그라운드에서 Vosk 마이크 스트림 실행

### STT 콜백 로직

Vosk가 음성을 해독할 때:

- 감지된 텍스트 인쇄
- "시간"이 포함된 경우 호출`speak_time()`



## 앱 실행

모든 것이 설치되면 다음을 수행하십시오.

```bash
python main.py
```

단계:

1. 시계가 나타납니다
2. **듣기**를 클릭하세요.
3. 말하세요:**“시간”**
4. 시계가 현재 시간을 알려줍니다.
5. 하단에 텍스트 애니메이션이 나타납니다.



## 문제 해결

### ❗ 마이크가 감지되지 않았습니다.

노력하다:

```bash
pip install sounddevice
```

또는 입력 장치를 선택하세요.

```python
sd.default.device = 1
```



### ❗ 음성이 감지되지 않았습니다.

**작은** 모델을 사용하세요. 큰 것에는 더 많은 CPU가 필요합니다.
명확하게 말하고 “LISTEN(듣기)”을 클릭한 후 1~2초 정도 기다리세요.



### ❗ TTS는 한 번만 작동합니다.

각 TTS 호출이 **새 엔진**을 생성하는지 확인하세요(제공된 코드에서 이미 완료됨).



## 전체 소스 코드

### 1. Windows DPI 인식

```python
import ctypes
try:
    ctypes.windll.user32.SetProcessDPIAware()
except:
    pass
```

- Windows의 **높은 DPI 화면**에 애플리케이션이 올바르게 표시되는지 확인합니다.
-에 싸서`try`다른 OS와의 호환성을 위해 차단합니다.


### 2. 수입품
```python
import pygame, math, datetime, sys, numpy as np, pyttsx3, threading, time
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json, os
```
- **pygame**: GUI 및 그래픽.
- **수학**: 시계 바늘의 삼각법.
- **datetime**: 시계 및 인사말의 현재 시간입니다.
- **numpy**: 인공적인 틱 소리를 생성합니다.
- **pyttsx3**: 텍스트 음성 변환 엔진.
- **스레딩**: 백그라운드에서 TTS/STT를 실행합니다.
- **사운드 장치 및 보스크**: 음성-텍스트 인식.
- **json & os**: Vosk 출력을 구문 분석하고 파일을 처리합니다.

### 3. 파이게임 초기화
```python
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PyShine Wall Clock")
```
- 사운드 재생을 위해 **파이게임** 및 **오디오 믹서**를 초기화합니다.
- **화면 크기** 및 창 **제목**을 설정합니다.

### 4. 상수와 색상
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
- 시계 문자판, 바늘, 버튼, 텍스트에 사용되는 **색상**을 정의합니다.

### 5. 시계 매개변수 및 글꼴
```python
center_x, center_y = WIDTH // 2, HEIGHT // 2
clock_radius = 150
font = pygame.font.SysFont('Arial', 24, bold=True)
date_font = pygame.font.SysFont('Arial', 20)
button_font = pygame.font.SysFont('Arial', 20, bold=True)
time_str_font = pygame.font.SysFont('Arial', 28, bold=True)
```
`center_x, center_y`: 시계의 중심.
`clock_radius`: 시계 문자판의 크기입니다.
- **숫자, 날짜, 버튼 텍스트 및 TTS 텍스트 표시**용 글꼴.

### 6. 틱 소리
```python
def create_tick_sound():
    ...
    tick_sound = pygame.sndarray.make_sound(sound_array)
    tick_sound.set_volume(0.5)
    return tick_sound
```
- NumPy를 사용하여 **1500Hz의 짧은 클릭**을 생성합니다.
- 외부 오디오 파일이 필요하지 않습니다.
- **매초마다 틱**을 재생하는 데 사용됩니다.

### 7. 듣기 버튼
```python
button_rect = pygame.Rect(WIDTH // 2 - 80, 80, 160, 50)
listening_active = False
def draw_button(mouse_pos):
    ...
```
- **화면에 버튼**을 그립니다.
- **호버링** 또는 **활성** 시 색상이 변경됩니다.
- **마이크 청취 상태**를 제어합니다.

### 8. 텍스트 입력 및 TTS
```python
def speak_time():
    ...
    threading.Thread(target=tts_func, args=(spoken_time_str,), daemon=True).start()
```
- 현재 시간을 기준으로 **인사말**을 결정합니다.
- **음성 텍스트** 형식: 예:`"Good afternoon\nIt's 03:25 PM now!"`
- **백그라운드 스레드에서 텍스트 음성 변환**을 시작합니다.
- **입력 애니메이션** 변수를 업데이트합니다.


### 9. 왁스 음성-텍스트 설정
```python
MODEL_PATH = "vosk-model-small-en-us-0.15"
vosk_model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, 16000)
```
- **오프라인 Vosk 모델**을 로드합니다.
- 인식기는 **오디오 바이트를 텍스트**로 변환합니다.
- **오프라인 음성 인식**을 보장합니다.

#### STT 콜백
```python
def stt_callback(indata, frames, time_data, status):
    ...
    if "time" in result_text.lower():
        speak_time()
```
- 마이크의 오디오를 처리합니다.
- 텍스트로 변환합니다.
- 트리거`speak_time()`**키워드 “time”**이 감지되면

### 10. 시계 그리기 기능
#### 시계 페이스
```python
def draw_clock_face():
    ...
```
- **바깥쪽 원, 시간 숫자, 분 틱**을 그립니다.
- **시간 단위**(두꺼움)와 **분 단위**(얇음)를 구분합니다.
#### 시계바늘
```python
def draw_clock_hands():
    ...
```
- 현재 시간을 기준으로 **시, 분, 초침**을 그립니다.
- **초마다 틱 소리**를 재생합니다.
- **중심 피벗** 원을 그립니다.
#### 날짜 표시
```python
def draw_date_display(now):
    ...
```
- **현재 날짜**와 **요일**을 표시합니다.

#### 타이핑 애니메이션

```python
def draw_spoken_time():
    ...
```
- 타이핑처럼 **인사말과 시간**을 순차적으로 보여줍니다.
- 커서 **깜박임**.
- **4초** 후에 자동으로 지워집니다.


### 11. 메인 루프

```python
def main():
    ...
```
- **이벤트** 처리:
- 그만두다
- ESC 키
- **듣기 버튼**을 마우스로 클릭하세요.
- 업데이트:
- **시계 페이스**
- **손**
- **날짜**
- **입력된 인사말**
- **듣기 버튼**
- **30FPS**에서 실행됩니다.
- **부드러운 애니메이션과 상호작용**을 보장합니다.

### 12. 진입점

```python
if __name__ == "__main__":
    main()
```
- 스크립트가 직접 실행될 때 **메인 루프**를 시작합니다.

### main.py

전체 작동 소스 코드는 다음과 같습니다.
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



**웹사이트:** https://www.pyshine.com
**저자:** 파이샤인
