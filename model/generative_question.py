from transformers import T5ForConditionalGeneration, T5Tokenizer
class QuestionAnswerGenerator:
    def __init__(self, model_name="Kais4rx/generative_question"):  
        self.model_name=model_name
    def load_model(self):
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
    def generate_question_answer(self,input_text):
        
        input = f"Generate a question and answer from the context: {input_text}"
    
        # Tokenize the input text
        input_ids = self.tokenizer(input, return_tensors="pt").input_ids

        # Generate the model output (question and answer pair)
        output_ids =self.model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)

        # Decode the generated ids to text
        generated_text =self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("Generated Text:", generated_text)

        return generated_text

if __name__ == "__main__":
    generator = QuestionAnswerGenerator()
    # Generate the question and answer

