#!/usr/bin/env python

##### Libraries

import json
import numpy as np
import os
import pandas as pd
import re

##### Setup

# Paths.
config_dir_path = '../../../config/'
data_dir_path = '../../../data/'
summary_dir_path = data_dir_path+'kenpom_summaries/'
snapshots_path = data_dir_path+'team_snapshots.json'
games_output_path = data_dir_path+'games.csv'

##### Main Execution

### KenPom

# Assemble KenPom data.
kenpom = None
kenpom_files = os.listdir(summary_dir_path)
# Traverse all files.
for kp_filename in kenpom_files:
	# Ignore non-final scores. Maybe we'll do something with these later.
	if not re.match(r'summary\d{2}.csv',kp_filename):
		continue
	# Get year.
	kp_year = int('20'+kp_filename[7:9])
	# Get data.
	kp_data = pd.read_csv(summary_dir_path+kp_filename)
	# Add year
	kp_data['year'] = kp_year
	# Save.
	if kenpom is None:
		kenpom = kp_data
	else:
		kenpom = kenpom.append(kp_data)

### Snapshots, Join

# Load snapshot data.
snapshots_dict = json.load(open(snapshots_path))

# Container for features.
snapshot_features = None

# Traverse snapshots.
for snap in snapshots_dict:
	# Get snapshot games.
	snap_games = pd.DataFrame(snap['games'])
	# Append team, year.
	snap_games['team'] = snap['team']
	snap_games['year'] = snap['year']
	# Join team information; prefix KenPom columns.
	snap_games = snap_games.merge(kenpom, left_on=['team','year'], right_on=['TeamName','year'])
	snap_games.drop('TeamName', axis=1, inplace=True)
	snap_games.columns = ['team_'+c if (c in kenpom.columns and c != 'year') else c for c in snap_games.columns]
	# Join opponent information; prefix KenPom columns.
	snap_games = snap_games.merge(kenpom, left_on=['opponent','year'], right_on=['TeamName','year'])
	snap_games.drop('TeamName', axis=1, inplace=True)
	snap_games.columns = ['opponent_'+c if (c in kenpom.columns and c != 'year') else c for c in snap_games.columns]
	if snapshot_features is None:
		snapshot_features = snap_games
	else:
		snapshot_features = snapshot_features.append(snap_games)

### Cleanup

# Integerize dummy variables.
for c, dt in zip(snapshot_features.columns,snapshot_features.dtypes):
	if dt == 'bool':
		snapshot_features[c] = snapshot_features[c].astype(np.int)

# Get location dummies.
location_dummies = pd.get_dummies(snapshot_features.location, prefix='location').astype(np.int)
# Clean up.
location_dummies.columns = [re.sub('[^a-zA-Z\\_]','',c) for c in location_dummies.columns]

# Append location dummies to right.
snapshot_features = pd.concat((snapshot_features, location_dummies), axis=1)

# Reorder columns.
# Store main column list.
main_snap_columns = [
		'year','date','team','opponent',
		'conference','conference_tournament','ncaa_tournament','other_tournament'
	]
outcome_columns = ['points_for','points_against','win']
# Add other columns.
kp_columns = [c for c in snapshot_features.columns if c.startswith('team_') or c.startswith('opponent_')]

# Reorder columns.
snapshot_features = snapshot_features[main_snap_columns+list(location_dummies.columns)+kp_columns+outcome_columns]

# Save.
snapshot_features.to_csv(games_output_path, index=False)
















