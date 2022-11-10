

def get_diff_as_json(filename):
    with open(filename) as f:
        lines = f.readlines()

    lines = [l.strip() for l in lines]
    changes ={}
    i = 0
    n = len(lines)

    current_file_name = None
    current_file_additions = []
    current_file_deletions = []

    while i < n:
        l = lines[i]
        if l.startswith("diff "):
            if current_file_name:
                changes[current_file_name] = {}
                changes[current_file_name]['addition'] = current_file_additions
                changes[current_file_name]['deletion'] = current_file_deletions
                current_file_additions = []
                current_file_deletions = []
            i+=2
            current_file_name = lines[i][6:]
            assert lines[i+1][6:] == current_file_name
            i+=2
        elif l.startswith('@@'):
            line_info = l.split('@@')[1].split()
            for info in line_info:
                key = info[0]
                info = info[1:]
                if ',' in info:
                    line_num, n_lines = [int(li) for li in info.split(',')]
                else:
                    n_lines = 1
                    line_num = int(info)
                if key == '+':
                    insert_index = line_num
                    n_append = n_lines
                elif key == '-':
                    delete_index = line_num
                    n_delete = n_lines
            i+=1
            j=0
            while j<n_delete and lines[i].startswith('-'):
                current_file_deletions.append(delete_index+j)
                j+=1
                i+=1
            assert n_delete == j
            j=0
            while j<n_append and lines[i].startswith('+'):
                current_file_additions.append(insert_index+j)
                j+=1
                i+=1
            assert n_append == j
        else:
            print(lines[i])
            i+=1

    if current_file_name:
        changes[current_file_name] = {}
        changes[current_file_name]['addition'] = current_file_additions
        changes[current_file_name]['deletion'] = current_file_deletions

    return changes
