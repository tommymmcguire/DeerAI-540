import os
from scripts.train_resnet import train_resnet
from scripts.resnet_eval import evaluate_resnet

def main():
    # Train the ResNet model
    train_resnet()
    
    # Evaluate the trained model
    evaluate_resnet()

if __name__ == "__main__":
    main()
