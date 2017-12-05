from bs4 import BeautifulSoup 
import requests
import pdb
import re
import pandas as pd 
from socket import error as SocketError 
import time 
import sys


def get_review(title):
	url = 'https://www.rottentomatoes.com/m/{}'.format(title)
	r = requests.get(url)
	bs = BeautifulSoup(r.text)
	rev_result = bs.findAll('p', {'class': 'critic_consensus tomato-info noSpacing superPageFontColor'})
	if len(rev_result) > 0:
		#pdb.set_trace()
		review = rev_result[0].text.split('\n')[2].strip()
		return review
	else:
		return -1

if __name__ == '__main__':
	manual_revs = []
	plots_df = pd.read_csv('tmdb-5000-movie-dataset/tmdb_5000_movies.csv')
	plots_df['review'] = ""
	for i, row in plots_df.iterrows():
		try: 
			title = row['original_title'].replace('-', '')
			title = title.replace(':', '')
			title = title.replace("'", '')
			title = title.replace(' ', '_')
			review = get_review(title)
			if review != -1:
				plots_df.loc[i, 'review'] = review
			else: 
				year = row['release_date'][:4]
				title_year = title + '_' + year
				review = get_review(title_year)
				if review != -1:
					plots_df.loc[i, 'review'] = review
				else:
					manual_revs.append(row['original_title'])
		except SocketError:
			print('ERROR')
			time.sleep(60)
			pass
		except AttributeError:
			print("SOMETHING IS WRONG")
			pass
		if i % 1000 == 0:
			print('pass', i)

	pdb.set_trace()
	out_plots_df = plots_df[['original_title', 'review', 'overview']].copy()
	out_plots_df.to_csv('scraped_reviews.csv')