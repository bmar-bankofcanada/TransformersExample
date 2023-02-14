import pandas as pd
import numpy as np
from datasets import Dataset
import json

class LabeledOilDatasets():

    def __init__(self,file_path,sep="|") -> None:
        self.df = pd.read_csv(file_path,sep=sep)

    def create_relevant_dataset(self) -> Dataset:
        self.df.dropna(inplace=True)
        self.df['labels'] = self.df['labels'].str.split("#", expand=False)
        labels = ['Prices Positive','Prices Negative','Supply Positive','Supply Negative','Demand Positive','Demand Negative','Future','Current','Intermediate','Not Relevant']
        for i in range(0,len(labels)):
            self.df[labels[i]] = np.zeros((len(self.df),1)).astype(int)
        # Kill label and turn it into the 0 and 1 columns
        for category in labels:
            self.df[category] = self.df['labels'].apply(lambda cat: 1 if category in cat else 0)
        self.df.drop(['labels'],axis=1,inplace=True)
        dataset = Dataset.from_pandas(self.df[['data','Not Relevant']])
        dataset = dataset.remove_columns('__index_level_0__') # Pandas clean up
        dataset = dataset.rename_column('Not Relevant','label')
        dataset = dataset.rename_column('data','text')
        dataset = dataset.train_test_split(test_size=0.1)
        return dataset
    

class OilJSONToDataset():

    def __init__(self,file_path) -> None:
        pass