import json
import os
from bot_tools.bot_funcs import Bot, pr_test_keys

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

if bot.is_user_trusted(event['comment']['user']['login']):
    bot.run_tests(pr_test_keys)
else:
    bot.warn_untrusted()
