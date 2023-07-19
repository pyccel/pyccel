import json
import os
from bot_tools.bot_funcs import Bot

if __name__ == '__main__':
    input_check_run_id = os.environ["GITHUB_CHECK_RUN_ID"]

    bot = Bot(pr_id = os.environ.get('PR_ID', 0), check_run_id = input_check_run_id, commit = os.environ["COMMIT"])

    if input_check_run_id == "":
        # Parse event payload from $GITHUB_EVENT_PATH variable
        # (documented here : https://docs.github.com/en/actions/learn-github-actions/variables#default-environment-variables)
        # The contents of this json file depend on the triggering event and are
        # described here :  https://docs.github.com/en/webhooks-and-events/webhooks/webhook-events-and-payloads
        with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as event_file:
            event = json.load(event_file)
        workflow_file = event["workflow"]
        test_key = os.path.splitext(os.path.basename(workflow_file))[0]
        posted = bot.create_in_progress_check_run(test_key)
    else:
        posted = bot.post_in_progress(int(os.environ['GITHUB_RUN_ATTEMPT']) > 1)

    run_id = posted['id']
    pr_id = bot.get_pr_id()
    sha = posted['head_sha']

    print(f"check_run_id={run_id}", sep='')
    print(os.environ["GITHUB_ENV"])
    with open(os.environ["GITHUB_ENV"], "a", encoding='utf-8') as f:
        print(f"CHECK_RUN_ID={run_id}", sep='', file=f)
        print(f"PR_ID={pr_id}", sep='', file=f)
        print(f"HEAD_SHA={sha}", sep='', file=f)
