from bs4 import BeautifulSoup 
import requests
import pdb
import re
import pandas as pd 
from socket import error as SocketError 
import time 
import sys

white_space_pattern = re.compile(r'/n')

start_i = int(sys.argv[1])
end_i = int(sys.argv[2])


for i in range(start_i, end_i):
	url = "http://www.imdb.com/title/tt{}/plotsummary?ref_=tt_stry".format(i)
	try: 
		r = requests.get(url)
		bs = BeautifulSoup(r.text)
		main = bs.find('div', {'class': "subpage_title_block"})
		title = main.findAll('a')[-1].text
		#print(title)
		#title = title[0].text
		plot = bs.findAll('ul', {'id': "plot-synopsis-content"})[0].text.strip()
		if not plot.startswith("It looks like we don't have a Synopsis"):
			print('\n\n', i)
			print(title)
			print(plot)
			movies['title'].append(title)
			movies['id'].append(i)
			movies['plot'].append(plot)
	except SocketError :
		print("\n\n\n ERROR \n\n\n")
		time.sleep(60)
		pass
	except AttributeError:
		pass


	if i % 5000 == 0 :
		file_name = "movies_{}.csv".format(i)
		df = pd.DataFrame(movies)
		df.to_csv(file_name)


final_df = pd.DataFrame(movies)
final_df.to_csv('final_movies_df.csv')

