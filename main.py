from data_transformer import LabeledOilDatasets
from train_relevance_model import RelevanceModel

def finetune():
    oil_dataset = LabeledOilDatasets(file_path="data/500_manually_labeled.csv")
    data = oil_dataset.create_relevant_dataset()
    model = RelevanceModel(
        transformer_model_name="microsoft/deberta-v3-base",
        dataset=data,
        id_label={0: "RELEVANT", 1: "NOT RELEVANT"},
        label_to_id={"RELEVANT": 0,"NOT RELEVANT":1}
    )
    model.fine_tune(output_dir="outputs/deBERTa_v3_base")
    
def categorize():
    pass


if __name__=="__main__":
    finetune()