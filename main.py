import os
from scripts.train_resnet import train_resnet
from scripts.resnet_eval import evaluate_resnet
from scripts.untrained_resnet import evaluate_untrained_resnet

def main():
    # Train the ResNet model
    train_resnet()
    
    # Evaluate the trained model
    evaluate_resnet()
    
    # Evaluate the untrained model
    evaluate_untrained_resnet()

if __name__ == "__main__":
    main()
