config = {
    'MONGO_ARGS': {
        'host': ['mongo0', 'mongo1', 'mongo2'],
        'port': 27017,
        #'username': USERNAME,
        #'password': PASSWORD,    
        'authSource': 'admin',
        'readPreference': 'nearest'
    },
    'GENDER_RECOGNITION': {
        'GENDERIZE_ENABLED': False,
        'GENDERAPI_ENABLED': True,
        #'GENDERAPI_TOKEN': XXXXXXXXXX,
        'HOST': 'localhost',
        'PORT': 5000
    },
    'NLP': {
        'MAX_BODY_LENGTH': 20000,
        'AUTHOR_BLOCKLIST': './GenderGapTracker/NLP/main/rules/author_blocklist.txt',
        'NAME_PATTERNS': './GenderGapTracker/NLP/main/rules/name_patterns.jsonl',
        'QUOTE_VERBS': './GenderGapTracker/NLP/main/rules/quote_verb_list.txt'
    }
}