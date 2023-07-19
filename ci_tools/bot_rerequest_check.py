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


    # Collect id from an issue_comment event with a created action
    pr_id = event['check_suite']['pull_requests'][0]['number']

    bot = Bot(pr_id = pr_id)

    print(event['check_suite']['check_runs_url'])
    print(event['check_suite']['url'])

    workflow_url = event['check_run']['details_url']

    workflow_id = workflow_url.split('/')[-1]

    bot.GAI.rerequest_run(workflow_id)
