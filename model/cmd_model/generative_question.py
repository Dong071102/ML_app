from transformers import T5ForConditionalGeneration, T5Tokenizer
model_name="Kais4rx/generative_question"
# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Check the model has been loaded
print(model)

def generate_question_answer(context, model, tokenizer):
    # Prepare the input text (context)
    input_text = f"Generate a question and answer from the context: {context}"
    
    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Generate the model output (question and answer pair)
    output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)

    # Decode the generated ids to text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text


# Example context from SQuAD or your own
context = """
The Amazon Rainforest, also known as the Amazon Jungle, is a vast tropical rainforest located in South America. It covers an area of approximately 5.5 million square kilometers (2.1 million square miles) and is the largest tropical rainforest in the world. The Amazon Rainforest spans nine countries: Brazil, Peru, Colombia, Venezuela, Ecuador, Bolivia, Guyana, Suriname, and French Guiana. It plays a crucial role in the global ecosystem, producing about 20% of the Earth's oxygen and acting as a significant carbon sink.

The Amazon River, the second-longest river in the world after the Nile, runs through the rainforest. The river is about 4,345 miles (7,062 kilometers) long and has the largest drainage basin of any river on Earth, covering over 7 million square kilometers. The Amazon River and its tributaries are home to a vast array of aquatic species, including the Amazon river dolphin, the piranha, and the anaconda.

The rainforest itself is home to an incredibly diverse range of plant and animal species. It is estimated that approximately 390 billion individual trees exist in the Amazon, belonging to around 16,000 different species. Some of the most famous trees include the rubber tree, the Brazil nut tree, and the kapok tree, which can grow up to 200 feet tall. The Amazon also contains a wide variety of medicinal plants, many of which have been used by indigenous peoples for centuries.

In addition to its biodiversity, the Amazon Rainforest is home to a rich cultural diversity. It is inhabited by numerous indigenous groups, including the Yanomami, Kayapo, and Ticuna peoples, who have lived in the region for thousands of years. These indigenous communities have their own unique languages, traditions, and ways of life, many of which are closely tied to the forest itself. They rely on the forest for food, medicine, and shelter, and have developed a deep spiritual connection with the land.

However, the Amazon Rainforest is facing significant environmental threats. Deforestation, primarily caused by logging, agriculture, and cattle ranching, has led to the loss of vast tracts of forest. In recent years, the rate of deforestation has increased, leading to concerns about the long-term impacts on biodiversity, climate change, and indigenous communities. The Brazilian government, along with international organizations, has taken steps to combat deforestation through conservation efforts, but challenges remain.

The Amazon also plays a key role in regulating the global climate. The rainforest acts as a massive carbon sink, absorbing large amounts of carbon dioxide from the atmosphere. This helps mitigate the effects of climate change by reducing the concentration of greenhouse gases. If deforestation continues at the current rate, it could lead to a tipping point where the Amazon shifts from being a carbon sink to a carbon emitter, releasing large amounts of stored carbon back into the atmosphere and further accelerating global warming.

In recent years, there has been growing awareness of the importance of protecting the Amazon Rainforest. Conservation groups, environmental activists, and indigenous rights organizations have worked together to raise awareness about the need to preserve the rainforest and its many unique species. They argue that protecting the Amazon is not only crucial for the well-being of the people and animals who live there, but for the health of the entire planet.

The Amazon's future remains uncertain, but there is hope that with continued efforts to combat deforestation and preserve its biodiversity, the rainforest can be protected for future generations. The survival of the Amazon Rainforest is not just important for South America, but for the entire world. It is a vital part of the global environmental system, and its protection is essential to the fight against climate change and the preservation of biodiversity on Earth.
"""

# Generate the question and answer
type=int(input('Enter your select:\n1. with example.\n2. with custom text. \nYour select: '))
if(type==2):
    context=str(input('Write your context: '))
generated_text = generate_question_answer(context, model, tokenizer)

print("Generated Text:", generated_text)
