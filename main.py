import requests 
from bs4 import BeautifulSoup

url = "https://www.film.ru/movies/memuary-ulitki"
req = requests.request('get',url)
soup = BeautifulSoup(req.text, 'html.parser')

f = open('test.txt', 'w', encoding="utf-8")
f.write(req.text)