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
title: "音声対応の挨拶付き掛け時計"
lang: ja
en_url: /Listening-and-Talking-clock/
zh_url: /zh/Listening-and-Talking-clock/

---
# 音声時刻と挨拶付き掛け時計

このチュートリアルでは、**Python ベースの壁掛け時計**を作成する方法を説明します。`pygame`**pyttsx3** と **Vosk** を使用した **text-to-speech** と **speech-to-text** です。アプリは「時間」という単語をリッスンし、現在の時刻と、現在の時刻に基づいた挨拶を返します。

<div class="ビデオコンテナ">
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

# 目次

1. [導入](#introduction)
2. [機能の概要](#features-overview)
3. [前提条件](#prerequisites)
4. [依存関係のインストール](#installing-dependencies)
    - [Windows](#windows)
    - [macOS](#macos)
    - [Linux](#linux)
5. [Speech-to-Text を理解する (Vosk)](#understanding-speech-to-text-vosk)
    - [音声合成が重要な理由](#why-speech-to-text-is- important)
    - [Vosk の仕組み — 理論 (簡略化)](#how-vosk-works--the- Theory-simplified)
    - [ワックスモデルの種類](#wax-model-types)
    - [言語モデルの入手先](#where-to-get- language-models)
    - [サポートされている言語](#supported-langages)
    - [初心者はどのモデルを使用するべきですか?](#what-model- should-beginners-use)
6. [テキスト読み上げ (pyttsx3) を理解する](#understanding-text-to-speech-pyttsx3)
    - [声を変える](#changing-voice)
    - [話す速度を変更する](#changing-speaking-speed)
7. [コードの内訳](#code-breakdown)
    - [クロックレンダリング](#クロックレンダリング)
    - [カチカチ音生成](#tick-sound-generation)
    - [タイピングアニメーション](#typing-animation)
    - [リッスンボタンの動作](#listen-button-behavior)
    - [STT コールバック ロジック](#stt-callback-logic)
8. [アプリの実行](#running-the-app)
9. [トラブルシューティング](#troubleshooting)
10. [完全なソースコード](#full-source-code)

## 導入

このプロジェクトは、**美しい掛け時計 GUI** を構築します。`pygame`— しかし、ひねりが加えられています:

* **時間を音声で読み上げる**ことができます。また、音声認識を使用して**時間を尋ねるのを聞く**ことができます。
* **「時間」** と言うと、アプリは **Vosk** を使用して音声を検出し、**pyttsx3** を使用して現在の時刻を読み上げ、画面の下部に滑らかな **入力アニメーション**を表示します。

## 機能の概要

### アナログ掛け時計

- 滑らかな秒針、分針、時針
- 日付と曜日の表示
- オプションのダークテーマ対応

### 内蔵カチカチ音

- NumPyを使用して人工的に生成
- 外部オーディオファイルは必要ありません

### 音声検出 (STT)

- Vosk オフライン音声認識を使用
- インターネットなしでも動作します
- 単純なキーワード（「時間」）を検出します

### テキスト読み上げ (TTS)

- pyttsx3 を使用します (オフライン)
- 自動的に読み上げます:
  *「こんにちは。現在午後 3 時 25 分です!」*

### タイピングアニメーション

- 挨拶と時間を表示します
- 滑らかに点滅するカーソル
- 数秒後に自動的にクリアされます

### 聞くボタン

- 継続的なマイクリスニングを切り替えます
- バックグラウンドスレッドで認識を実行します

## 前提条件

必要なのは以下だけです:

- Python 3.8+ (Python 3.12 を使用することをお勧めします)
- マイク
- 基本的な端末の使い方
- パッケージをインストールする機能

## 依存関係のインストール

### 窓

```bash
python -m venv py312
py312\Scripts\activate

pip install pygame pyttsx3 sounddevice vosk numpy
```

Vosk モデルをダウンロードします。
https://alphacephei.com/vosk/models

たとえばモデルを取得します [これ](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")

抽出して名前を変更します。

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

上記と同じ英語モデルをダウンロードします。
------------------------------------------

### Linux

```bash
python3 -m venv py312
source py312/bin/activate

pip install pygame pyttsx3 sounddevice vosk numpy
sudo apt update
sudo apt install -y libportaudio2 libportaudiocpp0 portaudio19-dev
```

## Speech-to-Text を理解する (Vosk)

Speech-to-Text (STT) は、話し言葉を書き言葉に変換するプロセスです。 Vosk は最も人気のあるオフライン STT エンジンの 1 つで、軽量で正確で、Python プロジェクトで使いやすいことで知られています。

以下は、チュートリアル、ドキュメント、または学習目的に適した詳細な説明です。

### Speech-to-Text が重要な理由

Speech-to-Text テクノロジは、次の理由から最新のソフトウェアに不可欠になっています。

#### ハンズフリー対話

ユーザーは音声を使用してアプリを制御でき、時計、アシスタント、および手が忙しいシナリオ (料理、運転など) に便利です。

#### アクセシビリティ

STT は、運動障害のあるユーザーや入力が難しいユーザーを支援します。

#### リアルタイムオートメーション

音声コマンドはイベントを即座にトリガーできます。例:
「タイマーを開始して」、「音楽を停止して」、「今何時ですか」。

#### 画面がなくても動作します

IoT デバイス、Raspberry Pi システム、または組み込みガジェットに役立ちます。

#### オフラインセキュリティ

Vosk は完全にオフラインで動作するため、音声データはクラウドに送信されず、プライバシーが強化されます。

### Vosk の仕組み — 理論 (簡略化)

Vosk は使い方が簡単に感じられますが、内部では本格的な音声処理理論が使用されています。わかりやすく初心者向けの説明は次のとおりです。

1. オーディオキャプチャ

* マイクは生の音声波形を記録します。
  ※これらの波は、時間の経過に伴う気圧の変化を表す単なる数値です。

2. 特徴抽出(MFCC)

* 生の音声は機械学習モデルにとって詳細すぎてノイズが多すぎます。
* Vosk は生のオーディオを MFCC 機能 (メル周波数ケプストラル係数) に変換します。

#### MFCC は以下を表します。

- 頻度分布
- 音量
- トーン
- 人間が音声として認識するパターン

*MFCC は、ニューラル ネットワークが理解できる音の指紋と考えてください。*

3. 音響モデル（ニューラルネットワーク）
   このモデルは MFCC の特徴を取り入れて音素を予測します。
   音の最小単位は次のとおりです。
   `k    a    t    ( = "cat" )`
   音響モデルは、数千時間の音声録音に基づいてトレーニングされています。
4. 言語モデル
   人間はランダムな音素シーケンスで話しません。
   したがって、言語モデルは、どの単語が意味をなすかを予測するのに役立ちます。

例えば：
音響モデルが次のようなものを検出した場合:
`d   t   a   m   p`
言語モデルは次のことをガイドします。
`→ "time"`
ちんぷんかんぷんの代わりに。
5. デコーダ
デコーダは以下を組み合わせます。

- 音響モデルからの予測
- 言語モデルからの確率
  and chooses the most likely final text output.
  Result: clear, readable text.

### 開発者が Vosk を好む理由

* 100% オフライン
* インターネットがないことは次のことを意味します:
  ✔ プライバシー
  ✔ 信頼性
  ✔ IoT またはフィールド環境に最適
* 低い CPU 使用率

実行対象:

- ラズベリーパイ
- 古いラップトップ
- ミッドレンジ PC
- 小型モデルも用意
- 一部のモデルは 50MB 未満です。
- 高速かつリアルタイム
- 小規模なハードウェアでも、即座に転写します。
- 多言語サポート

### ワックスモデルの種類

デバイスに基づいて選択できます。

#### 小型モデル

- <40MB
- 最速
- 精度が低い
- Raspberry Piや簡単なコマンドに最適
- この「音声時計プロジェクト」に最適

#### 中型モデル

- バランスの取れた精度 + スピード
- デスクトップまたはラップトップに適しています

#### 大型モデル

- 最高の精度
- CPU負荷が高くなる
- 単純な音声コマンドにはやりすぎ

### 言語モデルを入手できる場所

すべての公式モデルはこちらから:
https://alphacephei.com/vosk/models

### サポートされている言語

Vosk は以下をサポートします。

|言語 |モデル |

|英語 |`vosk-model-small-en-us-0.15`
|日本語 |`vosk-model-small-ja-0.22`
|中国語 |`vosk-model-small-cn-0.22`
|スペイン語 |`vosk-model-small-es-0.42`
|フランス語 |`vosk-model-small-fr-0.22`
|ヒンディー語 |`vosk-model-small-hi-0.22`

…その他にもたくさんあります。

### 初心者はどのモデルを使うべきですか？

**小さなモデル**を使用します。

- 速い
- CPU 使用率が低い
- ラズベリーパイに最適
- 単一単語のコマンドに対して十分な精度

小規模なモデル名の例:

`vosk-model-small-en-us-0.15`
`vosk-model-small-es-0.42`
`vosk-model-small-fr-0.22`

## Text-to-Speech を理解する (pyttsx3)

### 声を変える

コード内:

```python
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
```

### 話す速度を変更する

```python
engine.setProperty('rate', 150)
```

一般的な値:

- 120 (遅い)
- 150 (デフォルト)
- 180 (高速)

## コードの内訳

### クロックレンダリング

時計は手動で描画されます。

- 外側の円
- 時間番号
- 分刻み
- 時間に基づいて針を回転させる

### カチカチ音の発生

ロードする代わりに `.wav`、音声を生成します。

- 1500Hzのクリック
- 50ミリ秒の持続時間
- 指数関数的なフェード

NumPy のおかげで、外部ファイルをインポートしなくても時計は常に時を刻みます。

### タイピングアニメーション

挨拶は実際に入力しているように見えます。

- キャラクターが徐々に登場します
- カーソルが点滅します
- 4秒後、テキストは自動的に消去されます

### リッスンボタンの動作

- オン/オフを切り替えます
- 青 → アイドル状態
- 緑 → 聞いています
- Voskマイクストリームをバックグラウンドで実行します

### STT コールバック ロジック

Vosk が音声をデコードするとき:

- 検出されたテキストを印刷する
- 「時間」が含まれている場合は、電話してください。`speak_time()`

## アプリの実行

すべてがインストールされたら、次のようにします。

```bash
python main.py
```

手順:

1. 時計が表示されます
2. [**聞く**] をクリックします。
3. 話す:**「時間」**
   4.時計が現在の時刻を読み上げます
4. 下部にテキストアニメーションが表示されます

## トラブルシューティング

### ❗ マイクが検出されませんでした

試す：

```bash
pip install sounddevice
```

または、入力デバイスを選択します:

```python
sd.default.device = 1
```

### ❗ 音声が検出されませんでした

**小さい**モデルを使用してください。大きいものはより多くの CPU を必要とします。
はっきりと話して、「LISTEN」をクリックしてから 1 ～ 2 秒待ちます。

### ❗ TTSは一度だけ機能します

各 TTS 呼び出しで **新しいエンジン** が作成されるようにします (提供されたコードですでに実行されています)。

## 完全なソースコード

### 1. Windows DPI の認識

```python
import ctypes
try:
    ctypes.windll.user32.SetProcessDPIAware()
except:
    pass
```

- Windows の **高 DPI 画面**でアプリケーションが正しく表示されるようにします。
- に包まれた `try`他のOSとの互換性を保つためにブロックします。

### 2. 輸入

```python
import pygame, math, datetime, sys, numpy as np, pyttsx3, threading, time
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json, os
```

- **pygame**: GUI とグラフィックス。
- **数学**: 時計の針の三角法。
- **datetime**: 時計と挨拶の現在の時刻。
- **numpy**: 人工的なカチカチ音を生成します。
- **pyttsx3**: テキスト読み上げエンジン。
- **スレッド**: TTS/STT をバックグラウンドで実行します。
- **サウンドデバイスとvosk**: 音声テキスト認識。
- **json & os**: Vosk 出力を解析し、ファイルを処理します。

### 3. Pygameの初期化

```python
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PyShine Wall Clock")
```

- サウンド再生のために **Pygame** と **オーディオ ミキサー** を初期化します。
- **画面サイズ**とウィンドウの**タイトル**を設定します。

### 4. 定数と色

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

- 時計の文字盤、針、ボタン、テキストに使用される**色**を定義します。

### 5. クロックパラメータとフォント

```python
center_x, center_y = WIDTH // 2, HEIGHT // 2
clock_radius = 150
font = pygame.font.SysFont('Arial', 24, bold=True)
date_font = pygame.font.SysFont('Arial', 20)
button_font = pygame.font.SysFont('Arial', 20, bold=True)
time_str_font = pygame.font.SysFont('Arial', 28, bold=True)
```

`center_x, center_y`：時計の中心。
`clock_radius`: 文字盤のサイズ。

- **数字、日付、ボタン テキスト、および TTS テキスト表示**用のフォント。

### 6.カチカチ音

```python
def create_tick_sound():
    ...
    tick_sound = pygame.sndarray.make_sound(sound_array)
    tick_sound.set_volume(0.5)
    return tick_sound
```

- NumPy を使用して **短い 1500Hz クリック**を生成します。
- 外部オーディオファイルは必要ありません。
- **毎秒カチカチ**を再生するために使用されます。

### 7. 聞くボタン

```python
button_rect = pygame.Rect(WIDTH // 2 - 80, 80, 160, 50)
listening_active = False
def draw_button(mouse_pos):
    ...
```

- **画面上に**ボタン**を描画します。
- **ホバー**または**アクティブ**時に色が変わります。
- **マイクのリスニング状態**を制御します。

### 8. テキスト入力と TTS

```python
def speak_time():
    ...
    threading.Thread(target=tts_func, args=(spoken_time_str,), daemon=True).start()
```

- 現在の時間に基づいて **挨拶** を決定します。
- **音声テキスト**の形式: 例:`"Good afternoon\nIt's 03:25 PM now!"`。
- **バックグラウンド スレッドでテキスト読み上げ**を開始します。
- **入力アニメーション**変数を更新します。

### 9. Wax Speech-to-Text セットアップ

```python
MODEL_PATH = "vosk-model-small-en-us-0.15"
vosk_model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, 16000)
```

- **オフライン Vosk モデル**をロードします。
- 認識機能は **音声バイトをテキスト**に変換します。
- **オフライン音声認識**を保証します。

#### STT コールバック

```python
def stt_callback(indata, frames, time_data, status):
    ...
    if "time" in result_text.lower():
        speak_time()
```

- マイクからの音声を処理します。
- テキストに変換します。
- トリガー `speak_time()`**キーワード「時間」**が検出された場合。

### 10. 時計描画機能

#### 時計の文字盤

```python
def draw_clock_face():
    ...
```

- **外側の円、時間の数字、分の目盛り**を描画します。
- **時間目盛り** (太い) と **分目盛り** (細い) を区別します。

#### 時計の針

```python
def draw_clock_hands():
    ...
```

- 現在の時刻に基づいて **時、分、秒針**を描画します。
- **毎秒カチカチ音**を再生します。
- **中心ピボット**円を描画します。

#### 日付表示

```python
def draw_date_display(now):
    ...
```

- **現在の日付**と**曜日**を表示します。

#### タイピングアニメーション

```python
def draw_spoken_time():
    ...
```

- 入力するように**挨拶と時間を徐々に**表示します。
- カーソルが**点滅**します。
- **4 秒**後に自動的にクリアされます。

### 11. メインループ

```python
def main():
    ...
```

- **イベント**を処理します:
- やめる
- ESCキー
- **リスニングボタン**をマウスクリックします
- アップデート:
- **時計の文字盤**
- **手**
- **日付**
- **入力された挨拶**
- **聞くボタン**
- **30 FPS** で実行します。
- **スムーズなアニメーションとインタラクション**を保証します。

### 12. エントリーポイント

```python
if __name__ == "__main__":
    main()
```

- スクリプトが直接実行されると、**メイン ループ**が開始されます。

### main.py

完全に動作するソースコードは次のとおりです。
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

**ウェブサイト:** https://www.pyshine.com
**著者:** PyShine
