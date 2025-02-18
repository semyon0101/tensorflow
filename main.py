from concurrent.futures import as_completed
from requests_futures.sessions import FuturesSession
import requests  
from bs4 import BeautifulSoup
import json
import csv

with open('filmdata.csv', 'w', newline='', encoding="utf-8") as file:
    session = FuturesSession()
    session_=requests.session()

    writer = csv.writer(file)
    field = ["poster", "name", "dateCreated", "genre", "description", 'url']
    writer.writerow(field)

    i=0
    k=0
    while i<3500:
        i+=1
        response = session_.post("https://www.film.ru/a-z/movies/ajax?page="+str(i))
        soup = BeautifulSoup(response.text, 'html.parser')

        l=[]
        for a in BeautifulSoup(json.loads(soup.text)[1]['data'], "lxml").find_all('a',class_='wrapper_block_stack'):
            url = a.get('href')
            l.append(session.get("https://www.film.ru"+url))
        for future in as_completed(l): 
            try:
                soup_ = BeautifulSoup(future.result().text, 'html.parser')
                poster = soup_.find('meta', property="og:image").get('content')
                name = ''
                dateCreated= ''
                genre=[]
                description=[soup_.find('meta', property="og:description").get('content')]
                url_ = ""
                for text in soup_.find_all('script', type="application/ld+json"):
                    js = json.loads(text.text)
                    if js["@type"]=="Movie":
                        #print(json.dumps(json.loads(text.text), indent=4))
                        name = js['name']
                        dateCreated = js['dateCreated']
                        genre = js['genre']
                        description.append(js['description'])
                        url_ = js['url']
                writer.writerow((poster, name, dateCreated, genre, description, url_))

                k+=1            
                if k%10==0:
                    print(k)
            except:
                pass

'''
response = requests.post("https://www.film.ru/movies/doktor-nou")
if response.status_code == 200:
    print("Запрос выполнен успешно!")
    soup = BeautifulSoup(response.text, 'html.parser')
    print(soup)
    poster = soup.find('meta', property="og:image").get('content')
    name = ''
    dateCreated= ''
    genre=[]
    description=[soup.find('meta', property="og:description").get('content')]
    url_ = ""
    for text in soup.find_all('script', type="application/ld+json"):
        js = json.loads(text.text)
        if js["@type"]=="Movie":
            #print(json.dumps(json.loads(text.text), indent=4))
            name = js['name']
            dateCreated = js['dateCreated']
            genre = js['genre']
            description.append(js['description'])
            url_ = js['url']
    print(poster, name, dateCreated, genre, description, url_, sep="\n")
else:
    print(f"Ошибка! Статус код: {response.status_code}")
'''