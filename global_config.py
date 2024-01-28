import json 

config = json.load(open('plotResultsConf.json', 'r'))

numFile = config['num_files']
MK = config['MK']
ensembleFolder = config['ensemble_folder']