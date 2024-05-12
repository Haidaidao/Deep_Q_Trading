import json 

config = json.load(open('config.json', 'r'))

ensemble_type = config['ensemble_type']
num_files = config['num_files']
MK = config['MK']
ensemble_folder = config['ensemble_folder']
label_threshold = config['label_threshold']
epoch = config['epoch']
trend_type = config['trend_type']

def writeConfig(name, value):
    config = json.load(open('config.json', 'r'))
    config[name] = value
    json.dump(config, open('config.json', 'w'), indent=4)