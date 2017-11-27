from bs4 import BeautifulSoup 
import requests
import pdb
import re
import pandas as pd 


white_space_pattern = re.compile(r'/n')
movies = {'title': [], 'id': [], 'plot': []}

for i in range(4080000, 6000000):
	print('\n\n', i)
	url = "http://www.imdb.com/title/tt{}/plotsummary?ref_=tt_stry".format(i)
	r = requests.get(url)
	bs = BeautifulSoup(r.text)
	try: 
		main = bs.find('div', {'class': "subpage_title_block"})
		title = main.findAll('a')[-1].text
		#print(title)
		#title = title[0].text
		print(title)
		plot = bs.findAll('ul', {'id': "plot-synopsis-content"})[0].text.strip()
		if not plot.startswith("It looks like we don't have a Synopsis"):
			print(plot)
			movies['title'].append(title)
			movies['id'].append(i)
			movies['plot'].append(plot)
	except AttributeError:
		pass

	if i % 5000 == 0 :
		file_name = "movies4_{}.csv".format(i)
		df = pd.DataFrame(movies)
		df.to_csv(file_name)


final_df = pd.DataFrame(movies)
final_df.to_csv('final_movies_df.csv')

