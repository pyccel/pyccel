""" Script run as a reaction to a review left on a pull request.
"""
import json
import os
from bot_tools.bot_funcs import Bot


if __name__ == '__main__':
    # Collect id from a pull_request_review event with a created action
    pr_id = os.environ['pr_id']

    decision = os.environ['decision']

    bot = Bot(pr_id = pr_id)

    if decision == 'changes_requested':
        reviewer = os.environ['reviewer']
        bot.draft_due_to_changes_requested(reviewer)
    elif decision == 'approved':
        bot.mark_as_ready(following_review = True)
