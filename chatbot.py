from datasets import load_dataset
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip_itm import blip_itm
import torch.nn.functional as F
import time
from matplotlib import pyplot as plt 
import os


# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset "newyorker_caption_contest" provided from Kaggle
dataset = load_dataset("jmhessel/newyorker_caption_contest", 'explanation')


def load_demo_image(image_size,device, idx):
    raw_image = dataset['train']["image"][idx]  
    w,h = raw_image.size
    raw_image = raw_image.convert(mode='RGB') # convert 1 channel cartoon images to 3 channel
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image


def main():
    print("Welcome to ChatBot!\n")
    print("Importing model ... ")

    # Define image size 
    image_size = 384

    # Define Vision-Language Model for use. (BLIP)
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    
    # Ask user whether to use preprocessed data or not
    while True:
        to_preprocess = input("\nWill you preprocess the dataset from the beginning? (yes/no): ").lower()

        if to_preprocess =='yes':
            print("\nExtracting image features from the dataset ...")

            image_features = []

            # Use only trainset of the newyorker_caption_contest dataset for parsing --> Feature extraction requires 3484MiB memory
            start_time = time.time()
            total_len = len(dataset['train']['image'])
            for i in range(total_len):
                print(f"Iteration: {i+1}/{total_len}", end='\r')
                
                image = load_demo_image(image_size=image_size, device=device, idx=i)

                image_embedding = model.visual_encoder(image) 
                img_feature = F.normalize(model.vision_proj(image_embedding[:,0,:]),dim=-1)   

                image_features.append(img_feature.detach())

            # print(image_features[0].shape) # 1, 256
            print("\nFinished extraction (Time consumed: {:.4f}s)".format(time.time()-start_time))
            image_features = torch.vstack(image_features)

            # Save image features as a tensor
            fname = "./newyorker_caption_contest_sample.pt"
            torch.save(image_features, fname)
            break
        elif to_preprocess == 'no':
            print("Loading preprocessed dataset.")
            fname = "./newyorker_caption_contest.pt"
            image_features = torch.load(fname).to(device)
            break
        else:
            print("Invalid input. Please answer 'yes' or 'no'.")



    # Take query from user
    while True: # keep asking until user exits himself.
        input_prompt = input("\nUser input: ")

        # process user input with BLIP tokenizer and text feature extractor
        tokens = model.tokenizer(input_prompt, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(device) 

        text_embedding = model.text_encoder(tokens.input_ids, attention_mask = tokens.attention_mask,                      
                                            return_dict = True, mode = 'text')       
        text_feat = F.normalize(model.text_proj(text_embedding.last_hidden_state[:,0,:]),dim=-1)

        similarities = image_features @ text_feat.t()
        max_index = torch.argmax(similarities)

        print("The best match is: \n")
        
        plt.imshow(dataset['train']['image'][max_index], 'gray')
        plt.title(input_prompt)
        plt.show()
        plt.axis('off')
        plt.savefig("test.png", bbox_inches='tight')
        plt.savefig('./chatbot_results/{}.png'.format(input_prompt))

        choice = input("Do you want to continue? (yes/no): ")
        if choice.lower() != 'yes':
            print("\nExiting program.\nThank you.")
            break




if __name__ == "__main__":
    main()



