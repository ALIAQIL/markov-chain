from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

url = "https://en.wikipedia.org/wiki/Rabat"
req = Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
response = urlopen(req)
html = response.read()   
clean_text = ''.join(BeautifulSoup(html,"html.parser").stripped_strings)
print(clean_text)

with open('markov-chain/output.txt', 'w') as f:
    f.write(clean_text)