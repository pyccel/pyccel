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
    bot.create_in_progress_check_run(event["workflow"])
else:
    print(os.environ["GITHUB_CHECK_RUN_ID"])
    bot.post_in_progress()
