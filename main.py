from concurrent.futures import as_completed
from requests_futures.sessions import FuturesSession
from bs4 import BeautifulSoup
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import requests
def beautifulString(str):
    str = str.replace('\n', " ")
    str = str.replace('\t', ' ')
    while str.count('  '):
        str = str.replace('  ', ' ')
    return str

def openf(req):
    soup = BeautifulSoup(req.text, 'lxml')
    name = soup.find_all('h1')[0].string[6:]
    image = soup.find('div', class_ = 'poster movie').find('img').get("src")
    description = beautifulString(soup.find("div", class_='description').text)
    creationYear = soup.find('ul', class_ = 'attributes').find_all('li')[1].text
    return [req.url, name, image, description, creationYear]

api_url = 'https://www.yidio.com/redesign/json/browse_results.php'
params = {"type": "movie", "index": "0", "limit": "10000"}

with open('profiles1.csv', 'w', newline='', encoding="utf-8") as file:
    session = FuturesSession()

    writer = csv.writer(file)
    field = ["url", "name", "image", "description", "creationYear"]
    writer.writerow(field)
    for params['index'] in range(1,40000,10000): 
        print(params["index"])
        data = requests.get(api_url, params=params).json()
        l = []
        for el in data['response']:
            l.append(session.get(el["url"]))
        
        for future in as_completed(l):
            try:
                writer.writerow(openf(future.result()))
            except:
                print(future.result().url)
    
