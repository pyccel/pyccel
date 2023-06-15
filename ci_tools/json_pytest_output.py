import sys
import json


output_file = 'test_json_result.json'

def get_json_status():
    data = {}
    summary = {}
    data['title'] = 'pytest results'
    data['summary'] = 'text'
    return json.dumps(data)

with open(output_file, 'a') as f:
    print(get_json_status(), sep='', file=f)

