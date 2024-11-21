from tranformers import GPT2Tokenizer

def preprocess_story(story_path):
    with open(story_pqth, 'r'):
        text = f.read()

    tokenizer = GPT2Tokenizer.from_ptretrained('gpt2')
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i+1024] for i in range (0, len(tokens), 1024)]
    return chunks