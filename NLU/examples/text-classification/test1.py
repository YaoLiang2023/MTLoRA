from datasets import load_dataset

datasets = load_dataset("glue", "cola")
print("datasets : ", datasets)
datasets.to_csv("test")
# datasets.save_to_disk("test")