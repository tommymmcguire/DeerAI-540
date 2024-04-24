import os
from scripts.train_resnet import train_resnet
from scripts.resnet_eval import evaluate_resnet
from scripts.untrained_resnet import evaluate_untrained_resnet
from scripts.mean_model import run_mean_model
from scripts.classical_approach import ClassicalApproach

def main():
    # Train the ResNet model
    print("Training ResNet model...")
    train_resnet()
    
    # Evaluate the trained model
    print("Evaluating ResNet model...")
    evaluate_resnet()
    
    # Evaluate the untrained model
    print("Evaluating untrained ResNet model...")
    evaluate_untrained_resnet()
    
    # Evaluate the mean model
    print("Evaluating mean model...")
    run_mean_model()
    
    # Evaluate the classical approach
    print("Evaluating classical approach...")
    classical_approach = ClassicalApproach()
    classical_approach.run()
    

if __name__ == "__main__":
    main()
