from concurrent.futures import as_completed
from requests_futures.sessions import FuturesSession
from bs4 import BeautifulSoup
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
import time


def openf(req):
    
    soup = BeautifulSoup(req.text, 'lxml')
    img =  soup.find_all('img')[0]

    imageLink = img.get('src')
    title = img.get('alt')
    if len(soup.find_all('p'))==1:
        description = soup.find_all('p')[0].string 
    else:
        description = soup.find_all('p')[1].string 
    creationYear = soup.title.string.split(title)[1][2:6]
    return [req.url, imageLink, title, description, creationYear]


with open('profiles1.csv', 'w', newline='', encoding="utf-8") as file:
    session = FuturesSession()

    writer = csv.writer(file)
    field = ["ref", "imageLink", "title", "description", "creationYear"]
    writer.writerow(field)

    driver = webdriver.Chrome()
    i = 1
    while i!=416:
        print('page: '+str(i))
        driver.get('https://mubi.com/en/films?sort=popularity_quality_score&page='+str(i))
        ar = []
        while True:
            time.sleep(1)
            ar = driver.find_elements(By.TAG_NAME, 'a')
            if len(ar)==55:
                break
        l = []
        for a in ar:
            if a.get_attribute('data-testid')=="film-tile-link":

                l.append(session.get(a.get_attribute('href')))
        
        for future in as_completed(l):
            writer.writerow(openf(future.result()))
        
        time.sleep(1)
        i+=1
        
