""" Script run as a reaction to a review left on a pull request.
"""
import json
import os
from bot_tools.bot_funcs import Bot, pr_test_keys


if __name__ == '__main__':
    # Parse event payload from $GITHUB_EVENT_PATH variable
    # (documented here : https://docs.github.com/en/actions/learn-github-actions/variables#default-environment-variables)
    # The contents of this json file depend on the triggering event and are
    # described here :  https://docs.github.com/en/webhooks-and-events/webhooks/webhook-events-and-payloads
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as event_file:
        event = json.load(event_file)

    # If bot called explicitly (comment event)

    # Collect id from a pull_request_review event with a created action
    pr_id = event['pull_request']['number']

    decision = event['review']['state']

    bot = Bot(pr_id = pr_id)

    if decision == 'changes_requested':
        author = event['pull_request']['user']['login']
        reviewer = event['review']['user']['login']
        bot.draft_due_to_changes_requested(author, reviewer)
    else:
        bot.mark_as_ready(following_review = True)
