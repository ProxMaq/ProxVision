# For explanation of this code, you can visit this video: https://www.youtube.com/watch?v=YEUbMnMqJG0
import os
import requests # pip install requests #to sent GET requests
from bs4 import BeautifulSoup # pip install bs4 #to parse html(getting data out from html, xml or other markup languages)
from proxycrawl.proxycrawl_api import ProxyCrawlAPI # pip install proxycrawl #for more details visit: https://proxycrawl.com/

# download images from google search image URL
Google_Image = 'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&'

Image_Folder = 'Google Images' # Creating a folder to save the images and assigning to a variable for further use

def main():
    if not os.path.exists(Image_Folder):
        os.mkdir(Image_Folder)
    download_images()

def download_images():
    data = input('Enter your search keyword: ')
    num_images = int(input('Enter the number of images you want: '))
    
    print('Searching Images....')
    
    search_url = Google_Image + 'q=' + data #'q=' because its a query
    
    api = ProxyCrawlAPI({'token': 'Enter your ProxyCrawl Javascript token'}) #Enter your ProxyCrawl Javascript token, you will get this after registering in https://proxycrawl.com/ for trail and paid
    
    response = api.get(search_url, {'scroll': 'true', 'scroll_interval': '60', 'ajax_wait': 'true'}) #Parameters of ProxyCrawl
    if response['status_code'] == 200: #status code received from Google Site
        b_soup = BeautifulSoup(response['body'], 'html.parser') #Body for ProxyCrawl token, #html.parser is used to parse/extract features from HTML files
        results = b_soup.findAll('img', {'class': 'rg_i Q4LuWd'})
        
        #extract the links of requested number of images with 'data-src' attribute and appended those links to a list 'imagelinks'
    #allow to continue the loop in case query fails for non-data-src attributes        
        count = 0
        imagelinks= []
        for res in results:
            try:
                link = res['data-src']
                imagelinks.append(link)
                count = count + 1
                if (count >= num_images):
                    break
                
            except KeyError:
                continue
        
        print(f'Found {len(imagelinks)} images')
        print('Start downloading...')
    
        for i, imagelink in enumerate(imagelinks):
            response = requests.get(imagelink)
            
            # open each image link and save the file
            imagename = Image_Folder + '/' + data + str(i+1) + '.jpg'
            with open(imagename, 'wb') as file:
                file.write(response.content)
    
        print('Download Completed!')

if __name__ == '__main__':
    main()
