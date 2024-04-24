import os
from scripts.train_resnet import train_resnet
from scripts.resnet_eval import evaluate_resnet
from scripts.untrained_resnet import evaluate_untrained_resnet
from scripts.mean_model import run_mean_model
from scripts.classical_approach import ClassicalApproach

def main():
    # Train the ResNet model
    train_resnet()
    
    # Evaluate the trained model
    evaluate_resnet()
    
    # Evaluate the untrained model
    evaluate_untrained_resnet()
    
    # Evaluate the mean model
    run_mean_model()
    
    # Evaluate the classical approach
    classical_approach = ClassicalApproach()
    classical_approach.run()
    

if __name__ == "__main__":
    main()
