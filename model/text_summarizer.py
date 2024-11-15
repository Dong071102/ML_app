from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class TextSummarizer:
    def __init__(self, model_name="notHisuu/distilbart-xsum"):
        # Check if MPS or GPU is available, otherwise use CPU
        self.device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name=model_name
        print(f"Using device: {self.device}")
        
        # Load the model and tokenizer
    def load_model(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def summarize(self, input_text, max_length=150, num_beams=4):
        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)

        # Generate summary
        summary_ids = self.model.generate(inputs['input_ids'], max_length=max_length, num_beams=num_beams, early_stopping=True)

        # Decode the summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

# Example usage
if __name__ == "__main__":
    input_text = input("Enter the text to summarize: ")
