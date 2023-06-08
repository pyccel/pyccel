import sys
import json


output_file = 'test_json_result.json'

def get_json_status():
    data = {}
    summary = {}
    data['title'] = 'pytest results'
    summary['pytest 1'] = sys.argv[1]
    summary['pytest 2'] = sys.argv[2]
    summary['pytest 3'] = sys.argv[3]
    summary['pytest 4'] = sys.argv[4]
    summary['pytest 5'] = sys.argv[5]
    summary['pytest 6'] = sys.argv[6]
    data['summary'] = summary
    return json.dumps(data)

with open(output_file, 'a') as f:
    print(get_json_status(), sep='', file=f)

