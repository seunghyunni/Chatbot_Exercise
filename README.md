
# Repository for ChatBot Design Application exercise.

This repository holds the implementation for a chatbot that allows user to enter a message and returns the best matching image from the given image dataset:

Dataset:
<a href="https://nam04.safelinks.protection.outlook.com/?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Fjmhessel%2Fnewyorker_caption_contest&data=05%7C02%7Chwang229%40purdue.edu%7C788d3a9b767849d5c47308dc37cf8cd2%7C4130bd397c53419cb1e58758d6d63f21%7C0%7C0%7C638446611803407060%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=u6MBJr2tHVBccx5%2FURhBOCTZ%2BVffBcIFCA9yif6WW9w%3D&reserved=0">jmhessel/newyorker_caption_contest · Datasets at Hugging Face</a>

All codes are implemented in Python and tested using single NVIDIA RTX A5000 machine.

* Note that for text-to-image retrieval task, a pretrained Vision-Language model is adopted. Here, we use <a href="https://arxiv.org/abs/2201.12086">BLIP </a> [[blog](https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/)].
The official Pytorch implementation of the BLIP is used for implementing a ChatBot.
The code has been tested on PyTorch 1.10.

To import the BLIP model, the whole github repository[https://github.com/salesforce/BLIP/tree/main] of BLIP is cloned in this project. 
Note that only **'chatbot.py'** script in the folder is newly implemented. 

To run the model, please refer to the Requirements.

## Requirements

To install dependencies for running the application, please execute a following line.

```setup
pip install -r requirements.txt
```

## Dataset

Direct link to the 'newyorker_caption_contest' dataset is <a href="https://nam04.safelinks.protection.outlook.com/?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Fjmhessel%2Fnewyorker_caption_contest&data=05%7C02%7Chwang229%40purdue.edu%7C788d3a9b767849d5c47308dc37cf8cd2%7C4130bd397c53419cb1e58758d6d63f21%7C0%7C0%7C638446611803407060%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=u6MBJr2tHVBccx5%2FURhBOCTZ%2BVffBcIFCA9yif6WW9w%3D&reserved=0">here. </a>
The dataset is composed of 'train', 'test', and 'validation' sets. Each set contains 2,340, 131, and 130 images, respectively. 
**Here, we note that only 'train' set is used for parsing image.**

## Model

We used pretrained version of BLIP with ViT-B backbone, finetuned on the COCO dataset. The checkpoint can be found <a href="https://github.com/salesforce/BLIP/tree/main">here. </a>


## Code references and guidelines to new codes

Entire implementation of the BLIP model is borrowed from the official pytorch implementation of BLIP.
Using BLIP as a feature extractor for both image and text, the ChatBot which does text-to-image retrieval task is implemented. 

Note that running BLIP for extracting image features require 3500MiB GPU memory. Also, it takes approximately 2 seconds for extracting features from each image. 
Considering the expensive time for feature extraction process, I prepared a preprocessed data in advance, which is composed of feature vectors extracted from the whole 'train' set (2,340 images). The preprocessed data is in the 'BLIP' folder, under the name of **'newyorker_caption_contest.pt'**. 
However, user can always choose whether to use this preprocessed data or not. If user choose not to, then the feature extracting process will begin promptly during running the program.


## Testing

To run the application, please run:

```
python chatbot.py
```

1. Upon running, the program will ask whether to use preprocessed dataset or not. If answered 'yes', the program will begin preprocessing the dataset. If answered 'no', the program will skip the feature extraction part and ask user for input.

2. The user will be asked to input a message. Receiving a message, the program will promptly find a best matching image from the dataset and show the image. At the same time, the found image will be saved at the './chatbot_results' forder, under the name of user input. 

3. After returning the image, the program will ask the user to continue or not. If answered 'yes', the user is asked again for the new input. If answered 'no', the program will end. 
