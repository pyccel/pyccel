import os
import sys
from bot_tools.bot_funcs import Bot

def get_final_status(statuses : set):
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

bot = Bot(check_run_id = os.environ["check_run_id"], commit = os.environ["GITHUB_REF"])
bot.post_completed(set(sys.argv[2:]))
