from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load T5 model and tokenizer
model_name = "t5-small"  # You can use "t5-base" or "t5-large" for better performance
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def t5_summary(text, num_sentences=2):
    
    # Prepare input text for T5
    input_text = f"summarize: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    output_ids = model.generate(input_ids, max_length=num_sentences * 30, min_length=5, do_sample=False)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary
