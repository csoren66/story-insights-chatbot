from transformers import GPT2Tokenizer

def preprocess_story(story_path):
    with open(story_path, 'r') as f:
        text = f.read()
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i+1024] for i in range(0, len(tokens), 1024)]
    return chunks
