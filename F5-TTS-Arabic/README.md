---
language:
- ar
base_model:
- SWivid/F5-TTS
pipeline_tag: text-to-speech
tags:
- speech
- f5-tts
- arabic
- text-to-speech
- tts
datasets:
- MBZUAI/ClArTTS
- mozilla-foundation/common_voice_17_0
---
# F5-TTS: Fine-Tuned Arabic Speech Synthesis Model
## Update 1
the model can not produce high quality audio, the main goal of it is to provide a chekpoint to continue the fine tunning process from.
when using please set use_ema=False as it affects the quality.
Here is a notebook to try it quickly : https://colab.research.google.com/drive/1kX7HB05CouHa5A-4Wy0UPqMuW4APqDBr?usp=sharing
## Update 0
three checkpoints has bee added to the repo , the 380000 is the last , more data is needed to get better results so I will stop the fine tunning till getting more data and then i will proceed the fine tunning.
## Overview
This project fine-tunes the F5-TTS model for high-quality Arabic speech synthesis, incorporating regional diversity in pronunciation and accents. The fine-tuning process is ongoing, and temporary checkpoints are provided as progress updates. Future iterations will include improved models with enhanced accuracy and naturalness.

## Samples for now 
'''

1- "لكن على ما يبدو ان هناك تصاعد غير مسبوق للاحداث."

2- "لذلك يجب علينا الإتحاد فى وجه كل الصدامات التى قد تؤثر علينا."

3- "كان هناك الكثير من التحديات للوصول إلى الدقه المطلوبة."

'''
1- 

<audio controls src="https://cdn-uploads.huggingface.co/production/uploads/645098004f731658826cfe57/Co1vv5UnOffDEyPGY47li.wav"></audio>


2- 

<audio controls src="https://cdn-uploads.huggingface.co/production/uploads/645098004f731658826cfe57/jeKaMPd7f9P11aPCe5Y_0.wav"></audio>

3-

<audio controls src="https://cdn-uploads.huggingface.co/production/uploads/645098004f731658826cfe57/-c4gemoEcNX53CA21IheJ.wav"></audio>

## License
This model is released under the **CC BY-NC 4.0** license, which allows free usage, modification, and distribution for **non-commercial** purposes.

## Datasets
Training is based on the **MBZUAI/ClArTTS** so basically the model support MSA
## Model Information
- **Base Model:** SWivid/F5-TTS  
- **Current Status:** Ongoing fine-tuning (Temporary Checkpoints Available)  
- *(Final training parameters will be updated upon completion of fine-tuning.)*

## Usage Instructions
To use the fine-tuned Arabic model, follow these steps:


### Usage 
- **GitHub Repository:** Follow the [F5-TTS setup instructions](https://github.com/SWivid/F5-TTS), but replace the default model with the Arabic checkpoint and vocabulary files provided here.

## Contributions & Collaboration
This model is a **work in progress**, and community contributions are highly encouraged! Suggestions, improvements, and dataset contributions are welcome to refine its performance across different Arabic dialects.

### Recommendations for Better Results
- Use **clear reference audio** with minimal background noise.  
- Ensure **balanced audio levels** for improved synthesis quality.  
- Contributions in **dataset expansion** and **model evaluation** are highly valuable.
### Acknowledgment 
- This work is done using **Zewail City of science and technology machine**

  
If you have any questions or suggestions, feel free to reach out! 🚀