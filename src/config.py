import json

# Open the JSON file and load its contents
with open('../buzz.conf', 'r') as configfile:
    config = json.load(configfile)

    # Access the lists
    subjects = config['subjects']
    buzzwords = config['buzzwords']

    # Print the lists
    print("Subjects:", subjects)
    print("Buzzwords:", buzzwords)
