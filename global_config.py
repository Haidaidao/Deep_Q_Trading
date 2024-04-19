import json 

config = json.load(open('plotResultsConf.json', 'r'))

numFile = config['num_files']
MK = config['MK']
short = config['short']
middle = config['middle']
long = config['long']
ensembleFolder = config['ensemble_folder']
label_threshold = config['label_threshold']