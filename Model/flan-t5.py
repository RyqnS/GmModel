
""" Imports are mentioned here """
from IPython.display import clear_output
clear_output()

import csv
from datasets import load_dataset
from happytransformer import TTSettings
from happytransformer import TTTrainArgs
from happytransformer import HappyTextToText

t5Model = HappyTextToText("T5", "t5-base")

## EDIT--
#Downloading Data
train_dataset = load_dataset("jfleg", split='validation[:]')
eval_dataset = load_dataset("jfleg", split='test[:]')
#--

#Making csv?"
def generate_csv(csv_path, dataset):
    with open(csv_path, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile)
        writter.writerow(["input", "target"])
        for case in dataset:
     	    # Adding the task's prefix to input 
            input_text = "grammar: " + case["sentence"]
            for correction in case["corrections"]:
                # a few of the cases contain blank strings. 
                if input_text and correction:
                    writter.writerow([input_text, correction])

generate_csv("../Data/trainData.csv", train_dataset)
generate_csv("../Data/evalData.csv", eval_dataset)
#--

#Training
args = TTTrainArgs(batch_size=8)
t5Model.train("../Data/trainData.csv", args=args)

