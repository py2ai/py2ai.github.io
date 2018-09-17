---
layout: post
title: Speech controling the MS Power Point Presentation
author: Hussain A.
categories: [tutorial]
]
mathjax: true
summary: A simple AI application tutorial to control PPTX slides with speech using python
---

Hello friends, today i am going to present a very simple AI application to voice control the Microsoft PowerPoint Presentation.
The basic steps in making this application are:
1) Installation of the required libararies.
2) PySide based GUI to get the .pptx file and run it.
3) Listening to the voice command for two specific words such as; 'next' and 'bingo'. This is just for the demonstration purpose.
You can definetly choose your own magic words to move the slide. In my case saying the word 'next' will move forward the presentation
and `bingo` will move back the presentation.
4) For the speech recognition i am using SpeechRecognition API. 

Ok now lets start the first step, for this we need PySide.
