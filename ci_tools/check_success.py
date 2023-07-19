import sys

output_file = sys.argv[1]
statuses = set(sys.argv[2:])

def get_final_status(statuses : set):
    """
    Get the final status from results of each stage.

    Combine the status output of each stage of the
    test to determine the final combined status.

    Parameters
    ----------
    statuses : set
        The status of each stage.

    Returns
    -------
    str
        The final status.
    """
    if len(statuses) == 1:
        return statuses.pop()

    statuses.discard('skipped')
    if len(statuses) == 1:
        return statuses.pop()

    statuses.discard('success')
    if len(statuses) == 1:
        return statuses.pop()

    print(statuses)

    return statuses.pop()

with open(output_file, 'a') as f:
    print("status=",get_final_status(statuses), sep='', file=f)
