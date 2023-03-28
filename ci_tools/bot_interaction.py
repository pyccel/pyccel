import argparse
import json
import os
import sys
from git_evaluation_tools import leave_comment, get_status_json

#senior_reviewer = ['yguclu', 'ebourne']
senior_reviewer = ['ebourne']

test_keys = ['linux', 'windows', 'macosx', 'coverage', 'docs', 'pylint',
             'lint', 'spelling']

comment_folder = os.path.join(os.path.dirname(__file__), 'bot_messages')

def mark_as_ready(pr_id, outputs):
    """
    Mark the pull request as ready for review.

    Use the GitHub CLI to check if the PR is really ready for review.
    If this is the case then the correct flag is added and the draft
    status is removed.

    In order to be considered ready for review the PR must:
    - Have all tests passing
    - Have a non-trivial description

    Parameters
    ----------
    pr_id : int
        The number of the PR.
    """
    pass

def print_commands(pr_id):
    """
    List the available bot commands.

    Use the GitHub CLI to leave a comment on the pull request listing
    all the commands which the bot makes available.

    Parameters
    ----------
    pr_id : int
        The number of the PR.
    """

    bot_commands = ("This bot reacts to all comments which begin with `/bot`. This phrase can be followed by any of these commands:\n"
            "- `show tests` : Lists the tests which can be triggered\n"
            "- `run X` : Triggers the test X (X must be `all` or one of the values listed by `show tests`)\n"
            "- `mark as ready` : Adds the appropriate review flag and requests reviews. This command should be used when the PR is first ready for review, or when a review has been answered.\n"
            "- `commands` : Shows this list detailing all the commands which are understood")

    leave_comment(pr_id, bot_commands)

def welcome(pr_id, event):
    pass

bot_triggers = {'mark as ready': mark_as_ready,
                'commands' : print_commands}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Call the function to activate the bot')
    parser.add_argument('gitEvent', metavar='gitEvent', type=str,
                        help='File containing the json description of the triggering event')
    parser.add_argument('output', metavar='output', type=str,
                        help='File where the variables should be saved')
    parser.add_argument('run_id', metavar='output', type=int,
                        help='The id of the runner (used to identify the action page)')

    args = parser.parse_args()

    with open(args.gitEvent, encoding="utf-8") as event_file:
        event = json.load(event_file)

    outputs = {'run_linux': False,
               'run_windows': False,
               'run_macosx': False,
               'run_coverage': False,
               'run_docs': False,
               'run_pylint': False,
               'run_lint': False,
               'run_spelling': False,
               'HEAD': '',
               'REF': ''}

    if 'comment' in event and 'pull_request' in event['issue'] and event['comment']['body'].startswith('/bot'):
        pr_id = event['issue']['number']

        comment = event['comment']['body']
        command = comment.split('/bot')[1].strip()
        command_words = command.split()

        if command_words[0] == 'run':
            url = event['repository']['url']
            comment = "Running tests at {url}/actions/run/{run_id}\n"
            tests = command_words[1:]
            if tests == ['all']:
                tests = test_keys
            for t in tests:
                outputs[f'run_{t}'] = True
                comment += f"* :hourglass_flowing_sand: {t}"
            leave_comment(pr_id, comment)
        elif command == 'mark as ready':
            mark_as_ready(pr_id, outputs)
        elif command == 'show tests':
            with open(os.path.join(comment_folder, 'show_tests.txt')) as msg_file:
                comment = msg_file.read()
            leave_comment(pr_id, comment)
        else:
            print_commands(pr_id)

    elif 'pull_request' in event and not event['pull_request']['draft']:
        pr_id = event['number']
        event_name = 'pull_request'

        for k in test_keys:
            outputs[f'run_{k}'] = True

        if event['action'] == 'ready_for_review':
            mark_as_ready(pr_id, outputs)

    elif event['action'] == 'opened':
        new_user = event['pull_request']['author_association'] in ('COLLABORATOR', 'FIRST_TIME_CONTRIBUTOR')
        if new_user:
            with open(os.path.join(comment_folder, 'welcome_newcomer.txt')) as msg_file:
                comment = msg_file.read()
        else:
            with open(os.path.join(comment_folder, 'welcome_friend.txt')) as msg_file:
                comment = msg_file.read()
        with open(os.path.join(comment_folder, 'checklist.txt')) as msg_file:
            comment += msg_file.read()

        leave_comment(pr_id, comment)
    else:
        pr_id = None

    if pr_id is not None:
        outputs['HEAD'] = get_status_json(pr_id, 'baseRefName')
        outputs['REF'] = f'refs/pull/{pr_id}/merge'

    print(event)

    print(outputs)

    with open(args.output, encoding="utf-8", mode='a') as out_file:
        for o,v in outputs.items():
            print(f"{o}={v}", file=out_file)
