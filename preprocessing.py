import nltk
import string
import re
import wordninja
import ollama

with open('markov-chain/output.txt', 'r') as f:
    text = f.read()

text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text) #the problem there that there are no spaces between words, so we need to add spaces before capital letters
text = text.lower()
allowed_chars = set(string.ascii_lowercase + ' ')
text = ''.join(c for c in text if c in allowed_chars)
text = text.replace('\n', ' ')
text = ' '.join(text.split())   
stemmer = nltk.PorterStemmer()
text = ' '.join(stemmer.stem(word) for word in text.split()) #computations, computational, compute -> comput
text = " ".join(wordninja.split(text)) #it exists words like "thecat" and we need to split them into "the cat" without capital letters

#this method is not perfect, but it should be good enough for our purposes. We can always improve it later if we want to. The main goal here is to get a clean text that we can use to build our Markov chain.
# i was testing qwen3.5 last night so it was an opportunity to use it for this task
response = ollama.chat(model='qwen3.5:2b-q4_K_M', messages=[
  {
    'role': 'user',
    'content': f"Fix the spaces and formatting of this text, keep it lowercase: {text}",
  },
])

text = response['message']['content']

with open('cleaned_output.txt', 'w') as f:
    f.write(text)