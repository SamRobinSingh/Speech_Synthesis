# 🗣️ Speech Synthesis AI Framework


## 📝 Abstract

In the evolving landscape of Human-Computer Interaction (HCI), **natural voice-based communication** plays a pivotal role in bridging the gap between users and machines. This project presents a **conversational AI framework** that simulates human-like interactions using three critical modules: **Speech-to-Text (STT)**, **Text Generation**, and **Text-to-Speech (TTS)**.

The system begins by capturing the user's voice, accurately transcribes it into text using OpenAI’s **Whisper** model, processes it through a **language model (LLaMA 3.2)** to generate a relevant, context-aware response, and finally synthesizes the response back into human-like speech using **Kokoro TTS**, an open-weight text-to-speech engine.

This complete pipeline forms a real-time, end-to-end **speech-in → speech-out** solution that mimics human conversation, enabling its use in **virtual assistants**, **accessibility technologies**, **customer support**, and **educational applications**.

---

## 📌 Project Overview

The project is a **Speech-based AI Assistant** that understands your **spoken questions**, processes them with AI, and gives you a **spoken answer**.

It’s like talking to a smart assistant — you speak, it listens, thinks, and speaks back — all using AI.

> 🔄 It follows this pipeline:
**Speech Input (You talk) → Text → AI Response → Speech Output**

This is achieved through:
1. **Speech-to-Text** using **Whisper**
2. **Text Generation** using **LLaMA or GPT-style model**
3. **Text-to-Speech** using **Kokoro TTS**

---

## 🔬 Methodology

### Step-by-Step Breakdown:

1. 🎙️ **Speech Input**
   - The user speaks a query.
   - Audio is recorded as a `.wav` file.

2. 📝 **Speech-to-Text (STT)**
   - The recorded audio is passed to **Whisper**, a deep learning model for speech recognition.
   - It transcribes spoken words into text with high accuracy, even in noisy environments.

3. 🤖 **Text Generation**
   - The transcribed text (e.g., "What is AI?") is passed to a **Large Language Model** (like **LLaMA 3.2**).
   - The model generates a human-like text response based on the question.

4. 🔊 **Text-to-Speech (TTS)**
   - The AI-generated response is fed into the **Kokoro TTS** model.
   - It synthesizes expressive, clear speech from the response text.
   - The final output is played back to the user.

---

## 🚀 Features

- 🎙️ Real-time Speech Recognition with Whisper
- 🧠 Intelligent response generation using LLM (LLaMA 3.2)
- 🔊 Natural speech synthesis using Kokoro
- 📁 Modular design for easy extension and experimentation

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/SamRobinSingh/Speech_Synthesis.git
   cd Speech_Synthesis
2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
3. **Install Chocolatey (Windows only)**
   
   Follow instructions: https://chocolatey.org/install
   
6. **Install FFmpeg (required for Whisper)**
   ```bash
   choco install ffmpeg
7. **Run the main program**
   ```bash
   python main.py



