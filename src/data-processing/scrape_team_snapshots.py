#!/usr/bin/env python

### Libraries

import csv
import json
import os
import requests
import sys
import time
import urllib

### Setup

# Paths.
config_dir_path = '../../../config/'
data_dir_path = '../../../data/'
summary_dir_path = data_dir_path+'kenpom_summaries/'
export_dir_path = data_dir_path+'team_snapshots_raw/'

# Sleep time between fetches.
request_sleep_time = 5

# KenPom.com URL settings.
kp_domain = 'http://www.kenpom.com/'
kp_url_login = kp_domain+'handlers/login_handler.php'
kp_url_team  = kp_domain+'team.php?%s'

# KenPom login settings.
kp_credentials = json.load(open(config_dir_path+'kenpom.json'))
kp_headers     = json.load(open(config_dir_path+'kenpom_headers.json'))

### Main Execution

def main(argv=None):
	if argv is None:
		argv = sys.argv

	# Get year and teams. Don't worry about advanced error handling here.
	# If something doesn't work, we should be able to figure out why.
	year = sys.argv[1]
	teams = []
	with open(summary_dir_path+'summary'+year[2:]+'.csv', 'rb') as summary_csv:
		summary_reader = csv.reader(summary_csv, delimiter=',')
		for row in summary_reader:
			if row[0] != 'TeamName':
				teams.append(row[0])

	# Great, now we have teams. Traverse the whole list and download individual pages. Make sure to put several seconds in between.
	for t_i, team in enumerate(teams):
		# Filename.
		team_file = team+'.html'
		# Ensure the team data has not been fetched.
		if team_file in os.listdir(export_dir_path+year+'/'):
			continue
		
		# Fetch from KenPom.
		team_url = kp_url_team % urllib.urlencode({'team':team,'y':year})
		team_request = requests.get(team_url, headers=kp_headers)

		# Check status.
		if team_request.status_code != 200:
			print 'Last request returned status code %d.' % team_request.status_code
			print 'Response:'
			print team_request.text
			return 1

		# Write out.
		with open(export_dir_path+year+'/'+team_file,'w') as team_file:
			team_file.write(team_request.text)

		# Status update.
		if t_i % 20 == 0:
			print 'Just fetched %s (team %d).' % (team, t_i)
		# Sleep before next one.
		time.sleep(request_sleep_time)

	print 'Done.'

	# All done. Exit normally.
	return 0

if __name__ == '__main__':
	sys.exit(main())