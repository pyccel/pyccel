import sys

output_file = sys.argv[1]
statuses = set(sys.argv[2:])

def get_final_status(statuses : set):
    if len(statuses) == 1:
        return statuses.pop()

    statuses.pop('skipped')
    if len(statuses) == 1:
        return statuses.pop()

    statuses.pop('success')
    if len(statuses) == 1:
        return statuses.pop()

    print(statuses)

    return statuses.pop()

with open(output_file, 'a') as f:
    print("status=",get_final_status(statuses), sep='', file=f)
