from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Check if MPS is available
model_name="notHisuu/distilbart-xsum"
device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")
print(f"Using device: {device}")


# Load the fine-tuned model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Store the document as input_text and the summary as result
input_text = input('Input paragrap here:')
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
# Generate summary
summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
# Decode and print the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Generated summary:", summary)
