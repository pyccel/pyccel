""" Script that can be used to calculate the final combined status from results of multiple stages.
"""
import os
import sys
from bot_tools.bot_funcs import Bot

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
    statuses.discard('skipped')
    if len(statuses) == 0:
        return 'cancelled'

    elif len(statuses) == 1:
        return statuses.pop()

    statuses.discard('success')
    if len(statuses) == 1:
        return statuses.pop()

    print(statuses)

    return statuses.pop()

if __name__ == '__main__':
    bot = Bot(pr_id = os.environ["PR_ID"], check_run_id = os.environ["CHECK_RUN_ID"], commit = os.environ["COMMIT"])
    bot.post_completed(get_final_status(set(sys.argv[1:])))
