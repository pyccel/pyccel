""" Script run when a pull request is marked as ready for review.
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

    # Collect id from an issue_comment event with a created action
    pr_id = event['number']

    bot = Bot(pr_id = pr_id)

    if bot.is_user_trusted(event['sender']['login']):
        bot.request_mark_as_ready()
    else:
        bot.warn_untrusted()
