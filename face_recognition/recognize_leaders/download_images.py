import requests
from bs4 import BeautifulSoup

base_url = "https://www.gettyimages.com/photos/"



# getty images doesn't like automated bots hence i have to change the user agent
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.88 Safari/537.36"
}

html_image_class = "BLA_wBUJrga_SkfJ8won"

keywords = ['vladimir putin portrait','kim jong un portrait','joe biden portrait','donald trump portrait','erdogan portrait'] # change this as you want


LIMIT = 200
# install 500 images for each person
for keyword in keywords:
    name = keyword.replace('portrait','').strip()

    print(f"downloading images for {name}")
    response = requests.get(base_url+keyword.replace(' ','-'),headers=headers)
    soup = BeautifulSoup(response.text,'lxml')
    
    images = soup.find_all('img',class_=html_image_class)
    urls = [image['src'] for image in images[:LIMIT]]

    # download the images
    for i,url in enumerate(urls):
        response = requests.get(url,headers=headers)
        with open(f'assets/face_recognition/leaders/{name}/{name}-{i}.jpg','wb') as f:
            f.write(response.content)
        print(f'image {i} downloaded')

    print(f"downloading images for {name} done\n\n")
