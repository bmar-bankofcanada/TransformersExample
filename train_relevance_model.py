from datasets import Dataset 
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
import evaluate
import numpy as np

class RelevanceModel():
    '''
    class for training and using the Model on already labeled data
    '''
    def __init__(self,transformer_model_name,dataset,id_label,label_to_id) -> None:
        '''
        This class will download or use the local Model and Tokenizer 
        provided in the transformer_model_name string
        '''
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            transformer_model_name,
            num_labels=2,
            id2label= id_label,
            label2id= label_to_id
        )
        self.collorator = DataCollatorWithPadding(tokenizer = self.tokenizer)
        self.tokenized_dataset = dataset.map(self._preprocess_function,batched=True)

    def _compute_metrics(self,eval_pred):
        accuracy = evaluate.load('accuracy')
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    def _preprocess_function(self,examples):
        return self.tokenizer(examples['text'],truncation=True)
    
    def fine_tune(self,output_dir="output",lr=3e-5,batch_size=16,epochs=5,weight_decay=0.001):
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            report_to="none",
            load_best_model_at_end=True
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=self.collorator,
            compute_metrics=self._compute_metrics,
        )
        history = trainer.train()
        return history
