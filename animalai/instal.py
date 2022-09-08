from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

#key of API
key = "6e95be4c5b0f367ad474e346806d8f6b"
secret = "8fcafef1195d0f1c"
wait_time = 1

#decide the place of folder
animalname = sys.argv[1]
savedir = "./" + animalname

flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
  text = animalname,
  per_page = 400,
  media = 'photos',
  sort = 'relevance',
  safe_search = 1,
  extras = 'url_q, licence',
)

photos = result['photos']
#pprint(photos)

for i, photo in enumerate(photos['photo']):
  url_q = photo['url_q']
  filepath = savedir + '/' + photo['id'] + '.jpg'
  if  os.path.exists(filepath): continue
  urlretrieve(url_q,filepath)
  time.sleep(wait_time)