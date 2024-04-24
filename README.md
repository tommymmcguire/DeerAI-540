# Deer Age Predictor

## Description
This application utilizes machine learning and computer vision to predict the age of deer from images. It's designed for researchers, wildlife conservationists, and enthusiasts interested in studying deer populations and their age distribution. The main model utilizes a fine-tuned ResNET50 model trained on over 15,000 images of does of known age proived by the Caeser Kleberg Wildlife Research Institute in South Texas. The test and validation sets includes randomly selected images, with 2,175 and 3,909 images respectively. The total dataset includes 21,731 images and photographs of 16 individual does, raised and kept in an enclosed pen. The age group of these does are 2, 4, 7, 12, and 14 years old. 

## Results
Despite the limitations of this dataset, this model demonstrated impressive results with an MAE of 1.4709 and an RMSE of 2.3007. The model's predictions fall within +/- 2 years of the actual age 76.6% of the time. Grad-CAM analysis shows that these are some of the features that the model has learned from:



## File Structure
```
├── datasets/: Contains a small sample of the data.
  ├── train: Data used to train the model would go here.
  ├── val: Data used to perform validation would go here.
  ├── test: Data to test the models' performance.
├── /notebooks/: Includes Google Colab notebooks for exploratory analysis and initial testing of models and functions.
├── /scripts/: Contains all operational scripts required to run different parts of the pipeline.
  ├── train_resnet.py: Fine-tunes and trains the pretrained ResNET50 model on the deer data. 
  ├── resnet_eval.py: Evaluates the performance of the trained ResNET model.
  ├── untrained_resnet.py: Evaluates the performance of an untrained ResNET model for comparison.
  ├── mean_model.py: Implements the use of a naive mean model for comparison to the trained and untrained ResNET model.
  ├── classical_approach.py: Uses traditional machine learning techniques to determine the age of deer. 
├── main.py: Executes the entire pipeline, integrating various components.
├── app.py: Streamlit-based web application for user interaction.
├── requirements.txt: Includes all of the project's dependencies.
├── Makefile: Command to simplify capabilities such as installation, testing, and formatting (if applicable).
```

## Installation
To install Deer Age Predictor, follow these steps:
Clone the repository to your local machine:
```
git clone https://github.com/tommymmcguire/DeerAI-540.git
```
```
cd DeerAI-540
```

Install the required Python libraries:
```
make install
```
OR
```
pip install -r requirements.txt
```

## Usage 
To run the pipeline including all of the files in `scripts` simply run `main.py`

To run the files independently you can either:
1. edit main.py to only run the script of interest
2. cd into scripts and run the script of interest

Start the local server by running:
```
streamlit run app.py
```

This should automatically load the local server on your webpage. If not, click on the link provided in the terminal.

## Deployment
This app is currently deployed publicly on Streamlit. Go to https://agethedeer.streamlit.app/ to access the deployed application.