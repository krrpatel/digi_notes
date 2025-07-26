from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "vennify/t5-base-grammar-correction"
save_path = "./local_models/t5-grammar"

# Download from HuggingFace and save locally
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
