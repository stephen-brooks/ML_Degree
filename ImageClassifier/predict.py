import json
import argparse
import torch
from torch import nn
import numpy as np
from PIL import Image
from torchvision import transforms, models

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['model_class_idx']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    #Load image using Image.open
    pil_image = Image.open(image)
    #Transform image using similar methods as before
    transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
    #Convert to np array
    np_image = np.array(transform(pil_image))
    
    return np_image

def predict(img, model, topk,device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to("cpu")
    # Using GPU was causing issues
    # Create Tensor from array
    img_tens = torch.from_numpy(img).unsqueeze_(0)
    # Switch back to GPU if available
    model.to(device)
    # Save image to device
    img_tens = img_tens.to(device)
    # Run through forward pass with image
    output = model.forward(img_tens)
    # Get probabilities from likelihoods using SoftMax
    m = nn.Softmax(dim=1)
    preds = m(output)
    #Get topk probs and classes and convert to list for convenience, issues going from tensor to array
    top_probs, top_classes = preds.topk(topk)
    top_probs = top_probs.tolist()[0]
    top_classes = top_classes.tolist()[0]
    # Create a mapping for int to str
    mapping = {val: key for key, val in
                model.class_to_idx.items()
                }
    top_classes = [mapping [item] for item in top_classes]
    #Convert back to array for plotting
    return np.array(top_probs), np.array(top_classes)

def main():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument(dest = "image_path")
    parser.add_argument(dest = "checkpoint")
    parser.add_argument('--top_k', dest="top_k", action="store", default=5, type = int)
    parser.add_argument('--cat', dest="category_names", action="store", default="cat_to_name.json")
    parser.add_argument('--gpu', dest="gpu", action="store_true", default=False)
    args = parser.parse_args()
    
    device = torch.device("cuda" if args.gpu else "cpu")
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    model = load_checkpoint(args.checkpoint)
    image = process_image(args.image_path)
    probs, classes =  predict(image,model,args.top_k,device)
    # Map flower index to name
    flower_species = [cat_to_name[species] for species in classes] 
    print(dict(zip(flower_species, probs)))
    
if __name__ == '__main__': main()
