import requests
from bs4 import BeautifulSoup
req = requests.get("https://mubi.com/en/films?sort=popularity_quality_score")
src = req.text
soup = BeautifulSoup(src, 'lxml')
title = soup.title.string
print(title)