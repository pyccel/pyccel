""" Script run as a reaction to a review left on a pull request.
"""
import json
import os
import subprocess
from bot_tools.bot_funcs import github_cli, review_stage_labels, message_from_file, Bot

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

    if decision == 'changes_requested':
        author = event['pull_request']['user']['login']
        reviewer = event['review']['user']['login']
        subprocess.run([github_cli, 'pr', 'ready', pr_id, '--undo'], check=False)
        subprocess.run([github_cli, 'pr', 'comment', pr_id, '--body',
            message_from_file('set_draft_changes.txt').format(author=author, reviewer=reviewer)],
            check=True)
        for l in review_stage_labels:
            subprocess.run([github_cli, 'pr', 'edit', pr_id, '--remove-label', l], check=False)
    elif decision == 'approved':
        p = subprocess.run([github_cli, 'pr', 'view', pr_id, '--json', 'headRepositoryOwner,headRepository'],
                check=True, capture_output=True, text=True)
        repo_info = json.loads(p.stdout)
        repo = f"{repo_info['headRepositoryOwner']['login']}/{repo_info['headRepository']['login']}"
        if p.stdout == 'pyccel/pyccel':
            bot = Bot(pr_id = pr_id)
            bot.mark_as_ready(following_review = True)
        else:
            subprocess.run([github_cli, 'pr', 'comment', pr_id, '--body',
                '/bot approved pr fork'], check=True)
