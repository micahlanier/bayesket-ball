#!/usr/bin/env python

### Libraries

from bs4 import BeautifulSoup as bs
import datetime as dt
import json
import os
import re

### Setup

# Paths to snapshots, schedules.
snapshot_dir_path      = '../../../data/team_snapshots_raw/'
snapshot_proc_dir_path = '../../../data/team_snapshots_processed/'

# Processed file paths.
output_path = '../../../data/team_snapshots.json'

### Main Execution

# Get years for which we have data.
snapshot_years = [int(p) for p in os.listdir(snapshot_dir_path) if re.match(r'\d{4}',p)]

# Container to store all snapshots.
snapshots = []

# Traverse all year directories.
for y in snapshot_years:

	# Get all team files.
	raw_y_dir = snapshot_dir_path+str(y)+'/'
	team_snaps = [p for p in os.listdir(raw_y_dir) if p[-5:] == '.html']

	# Table format changed slightly in 2010. Store a constant to adjust.
	rk_const = 0 if y <= 2010 else 1

	# Traverse them.
	for t in team_snaps:
		# Get data.
		t_data = bs(open(raw_y_dir+t))
		# Get schedule table.
		sched_table = t_data.findAll('table')[1]

		# Clean team name.
		team_name = t[:-5]

		# Get record information.
		rank_info = t_data.findAll('span', {'class':'rank'})[1].text
		wins = int(rank_info[1:rank_info.index('-')])
		losses = int(rank_info[rank_info.index('-')+1:-1])

		# Get conference information.
		conf = t_data.find('div', {'id':'title-container'}).find('span', {'class':'otherinfo'}).find('a').text

		# Containers to store information.
		t_info = {
			'team': team_name,
			'year': y,
			'conference': conf,
			'games': [],
			'wins': wins,
			'losses': losses,
			'conference_wins': 0,
			'conference_losses': 0,
			'ncaa_tournament_wins': 0
		}

		# Set up some booleans to help us determine kinds of games.
		conference = False
		conference_tournament = False
		ncaa_tournament = False
		other_tournament = False
		last_conf_record = ''

		# These tables are all well-formed. Traverse the appropriate row range of each and store data.
		sched_rows = sched_table.findAll('tr')[1:-1]
		for sr in sched_rows:
			# Look for class in row. If exists, this is a valid game.
			if sr.has_attr('class'):
				# Get information that requires pre-processing.
				# Get score.
				score_info = sr.findAll('td')[rk_const+3].text
				score = score_info[score_info.find(', ')+2:]
				# Win info.
				win = score_info[0] == 'W'
				# Find score accordingly.
				points_greater = int(score[:score.find('-')])
				points_lesser  = int(score[score.find('-')+1:])
				points_for     = points_greater if win else points_lesser
				points_against = points_lesser if win else points_greater
				# Get date.
				date_text = sr.find('td').text
				date_text = date_text[date_text.index(' ')+1:]
				if date_text.startswith('Nov') or date_text.startswith('Dec'):
					date_text += ' '+str(y-1)
				else:
					date_text += ' '+str(y)
				date_clean = dt.datetime.strptime(date_text,'%b %d %Y').date().strftime('%Y-%m-%d')
				# Conference info.
				conf_record = sr.findAll('td')[-1].text
				conference = conf_record != last_conf_record and re.match(r'\d+-\d+',conf_record) is not None
				last_conf_record = conf_record
				# Append game info to games.
				t_info['games'].append({
					'date': date_clean,
					'opponent': sr.findAll('td')[rk_const+2].text,
					'location': sr.findAll('td')[rk_const+6].text,
					'conference': conference or conference_tournament,
					'conference_tournament': conference_tournament,
					'ncaa_tournament': ncaa_tournament,
					'other_tournament': other_tournament,
					'win': win,
					'points_for': points_for,
					'points_against': points_against
				})
				# Update win counts.
				t_info['conference_wins'] += (conference or conference_tournament) and win
				t_info['conference_losses'] += (conference or conference_tournament) and not win
				t_info['ncaa_tournament_wins'] += ncaa_tournament and win
			else:
				# If not a game row we can at least get information about the row we've encountered.
				# We can reason about what order certain things should come in.
				if sr.text == 'NCAA Tournament':
					conference_tournament = False
					ncaa_tournament = True
				elif sr.text == 'Postseason':
					conference_tournament = False
					other_tournament = True
				elif sr.text.endswith('Conference Tournament'):
					conference_tournament = True

		# Whew, now we have season info. Append to year snaps.
		snapshots.append(t_info)

# Write out all snapshot data.
json.dump(snapshots, open(output_path,'w'))
