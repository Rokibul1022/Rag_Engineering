from datasets import load_dataset

ds = load_dataset("microsoft/ms_marco", "v1.1")
ds.save_to_disk("./ms_marco/v1.1")

print(ds)
print(ds["train"][0])