import json
import os
from bot_tools.bot_funcs import Bot

bot = Bot(check_run_id = os.environ["GITHUB_CHECK_RUN_ID"], commit = os.environ["GITHUB_REF"])

if os.environ["GITHUB_CHECK_RUN_ID"]=="":
    # Parse event payload from $GITHUB_EVENT_PATH variable
    # (documented here : https://docs.github.com/en/actions/learn-github-actions/variables#default-environment-variables)
    # The contents of this json file depend on the triggering event and are
    # described here :  https://docs.github.com/en/webhooks-and-events/webhooks/webhook-events-and-payloads
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as event_file:
        event = json.load(event_file)
    workflow_file = event["workflow"]
    test_key = os.path.splitext(os.path.basename(workflow_file))[0]
    run_id = bot.create_in_progress_check_run(test_key)
else:
    print(os.environ["GITHUB_CHECK_RUN_ID"])
    bot.post_in_progress()
    run_id = os.environ["GITHUB_CHECK_RUN_ID"]

print(f"check_run_id={run_id}", sep='')
print(os.environ["GITHUB_ENV"], "a")
with open(os.environ["GITHUB_ENV"], "a") as f:
    print(f"check_run_id={run_id}", sep='', file=f)
