"""
This module contains all function and logic responsible for controlling the bot which reacts to PR creation and progress.
"""
import argparse
import json
import os
import sys
from git_evaluation_tools import leave_comment, get_status_json, get_job_information
from git_evaluation_tools import check_previous_comments, set_ready, set_draft, get_review_status
from git_evaluation_tools import check_previous_contributions, add_labels, remove_labels, get_labels
from git_evaluation_tools import get_previous_pr_comments

senior_reviewer = ['yguclu', 'EmilyBourne']
trusted_reviewers = ['yguclu', 'EmilyBourne', 'ratnania', 'saidctb', 'bauom']

#pr_test_keys = ['linux', 'windows', 'macosx', 'coverage', 'docs', 'pylint',
#             'lint', 'spelling']
pr_test_keys = ['docs', 'pylint', 'spelling']

review_labels = ('needs_initial_review', 'Ready_for_review', 'Ready_to_merge')

comment_folder = os.path.join(os.path.dirname(__file__), 'bot_messages')

def get_run_url(event):
    """
    Get the URL of the workflow run.

    Use the event information to calculated the URL
    where the results of the workflow can be viewed.

    Parameters
    ----------
    event : dict
        The event payload of the GitHub workflow.

    Results
    -------
    str : The URL.
    """
    url = event['repository']['html_url']
    run_id = event['run_number']
    return f"{url}/actions/runs/{run_id}"

def run_tests(pr_id, tests, outputs, event):
    """
    Run the requested tests and leave a comment on the PR.

    Update the outputs to ensure that the requested tests are run.
    Leave a comment on the PR using the GitHub CLI to indicate the relevant
    commit, action and the test status.

    Parameters
    ----------
    pr_id : int
        The number of the PR.

    tests : list of strs
        The tests requested.

    outputs : dict
        The dictionary containing the output of the bot.

    event : dict
        The event payload of the GitHub workflow.

    Returns
    -------
    bool
        Indicates if any tests were triggered.
    """
    # Leave a comment to link to the run page
    ref_sha = get_status_json(pr_id, 'headRefOid')
    url = get_run_url(event)

    unrecognised = []
    running = set()

    # Modify the flags to trigger the tests
    if tests == ['pr_tests']:
        tests = pr_test_keys
        running = set(pr_test_keys)
    for t in tests:
        key = f'run_{t}'
        if key in outputs:
            outputs[key] = True
            running.add(t)
        else:
            unrecognised.append(t)

    if outputs['run_coverage']:
        outputs['run_linux'] = True
        running.add('linux')

    running_tests = bool(running)
    print(unrecognised)
    print(tests)

    if not running_tests:
        comment = "The requested tests were not recognised. I detected:\n"
        comment += '\n'.join(f'- "{u}"' for u in unrecognised)
        comment += "\n\n"
        comment += message_from_file('show_tests.txt')
    else:
        comment = f"Running tests on commit {ref_sha}, for more details see [here]({url})\n"
        comment += '\n'.join(f'- :hourglass_flowing_sand: {r}' for r in running)
        if unrecognised:
            comment += "\n\nThe following additional tests were not recognised:\n"
            comment += '\n'.join(f'- "{u}"' for u in unrecognised)
            comment += "\n\n"
            comment += "To see a list of all test names, please use `/bot show tests`"

    leave_comment(pr_id, comment)

    outputs['status_url'] = event['repository']['statuses_url'].format(sha=ref_sha)

    return running_tests

def check_review_stage(pr_id):
    """
    Find the review stage.

    Use the GitHub CLI to examine the reviews left on the pull request
    and determine the current stage of the review process.

    Parameters
    ----------
    pr_id : int
        The number of the PR.

    Results
    -------
    bool : Indicates if the PR is ready to merge.

    bool : Assuming the PR is not ready to merge, indicates if the PR is
            ready for a review from a senior reviewer.

    requested_changes : List of authors who requested changes.

    reviews : Summary of all reviews left on the PR.
    """
    reviews, _ = get_review_status(pr_id)
    senior_review = [r for a,r in reviews.items() if a in senior_reviewer]

    other_review = [r for a,r in reviews.items() if a not in senior_reviewer]

    ready_to_merge = any(r.state == 'APPROVED' for r in senior_review) and not any(r.state == 'CHANGES_REQUESTED' for r in senior_review)

    ready_for_senior_review = any(r.state == 'APPROVED' for r in other_review) and not any(r.state == 'CHANGES_REQUESTED' for r in other_review)

    requested_changes = [a for a,r in reviews.items() if r.state == 'CHANGES_REQUESTED']

    return ready_to_merge, ready_for_senior_review, requested_changes, reviews

def set_review_stage(pr_id):
    """
    Set the flags for the review stage.

    Determine the current stage of the review process from the state
    of the reviews left on the pull request. Leave the flag indicating
    that stage and a message encouraging reviews.

    Parameters
    ----------
    pr_id : int
        The number of the PR.
    """
    ready_to_merge, ready_for_senior_review, requested_changes, reviews = check_review_stage(pr_id)
    author = get_status_json(pr_id, 'author')['login']
    labels = get_labels(pr_id)
    if ready_to_merge:
        add_labels(pr_id, ['Ready_to_merge'])
    elif ready_for_senior_review:
        if 'Ready_for_review' not in labels:
            add_labels(pr_id, ['Ready_for_review'])
            if any(r in requested_changes for r in senior_reviewer):
                if all(l not in review_labels for l in labels):
                    requested = ', '.join(f'@{r}' for r in requested_changes)
                    message = message_from_file('rerequest_review.txt').format(
                                                    reviewers=requested, author=author)
                    leave_comment(pr_id, message)
            else:
                names = ', '.join(f'@{r}' for r in senior_reviewer)
                approved = ', '.join(f'@{r}' for r in reviews.values() if r.state == 'APPROVED')
                message = message_from_file('senior_review.txt').format(
                                reviewers=names, author=author, approved=approved)
                leave_comment(pr_id, message)
    else:
        if all(l not in review_labels for l in labels):
            add_labels(pr_id, ['needs_initial_review'])
            if requested_changes:
                if all(l not in review_labels for l in labels):
                    requested = ', '.join(f'@{r}' for r in requested_changes)
                    message = message_from_file('rerequest_review.txt').format(
                                                    reviewers=requested, author=author)
                    leave_comment(pr_id, message)
            else:
                message = message_from_file('new_pr.txt').format(author=author)
                leave_comment(pr_id, message)

def mark_as_ready(pr_id):
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
    data = get_job_information(event['run_number'])

    job_data = [j for j in data if j['name'] not in ('Bot', 'CleanUpBot')]

    failures = [j['name'] for j in job_data if j['conclusion'] in ('cancelled', 'failed')]

    with open(os.path.join(os.path.dirname(__file__), 'codacy.json'), encoding="utf-8") as codacy_results:
        codacy_results = json.loads(codacy_results.read())
    print(codacy_results)

    if failures:
        set_draft(pr_id)
        message = message_from_file('set_draft_failing.txt')
        for f in failures:
            message += f'- {f}\n'
        leave_comment(pr_id, message)
    else:
        set_ready(pr_id)

        set_review_stage(pr_id)

    return 'failure' if failures else 'success'

def message_from_file(filename):
    """
    Get the message saved in the file.

    Reads the contents of the file `filename`, located in the
    folder ./bot_messages/. The extracted string is returned for
    use as a comment on a PR.

    Parameters
    ----------
    filename : str
        The name of the file to be read

    Results
    -------
    str : The message to be printed.
    """
    with open(os.path.join(comment_folder, filename), encoding="utf-8") as msg_file:
        comment = msg_file.read()
    return comment

def update_test_information(pr_id, event):
    """
    Update the PR with the information about the tests.

    Use the GitHub CLI to check the results of the tests. If the last
    comment made by the bot was for this set of tests then this comment
    is updated to include the test results. Otherwise a new comment is
    added with the results.

    Parameters
    ----------
    pr_id : int
        The number of the PR.

    event : dict
        The event payload of the GitHub workflow.
    """
    messages, last_message, _ = check_previous_comments(pr_id)

    data = get_job_information(event['run_number'])

    url = get_run_url(event)
    relevant_messages = [m for m in messages if url in m]
    ref_sha = relevant_messages[0].split()[4]
    comment = f"Ran tests on commit {ref_sha} for more details see [here]({url})\n"
    passed = True
    for job in data:
        conclusion = job['conclusion']
        name = job['name']
        if conclusion == 'skipped' or name in ('Bot', 'CleanUpBot'):
            continue
        job_passed = (conclusion == 'success')
        icon = ':heavy_check_mark:' if job_passed else ':x:'
        comment += f"- {icon} {name}\n"
        passed &= job_passed

    leave_comment(pr_id, comment, url in last_message)

    return 'success' if passed else 'failure'

def start_review_check(pr_id, event, outputs):
    """
    Check if the review is as ready as the author thinks.

    Use the GitHub CLI to check if the PR has a meaningful
    description (at least 3 words). If this is the case then the
    tests are triggered to determine the final draft status.

    Parameters
    ----------
    pr_id : int
        The number of the PR.

    event : dict
        The event payload of the GitHub workflow.

    outputs : dict
        The dictionary containing the output of the bot.
    """

    description = get_status_json(pr_id, 'body')
    words = description.split()

    if len(words) < 3:
        leave_comment(pr_id, message_from_file('set_draft_no_description.txt'))
        set_draft(pr_id)
    else:
        outputs['cleanup_trigger'] = 'request_review_status'
        run_tests(pr_id, ['pr_tests'], outputs, event)

def flagged_as_trusted(pr_id, user):
    """
    Check if the user has been flagged as trusted.

    Look through the previous comments on the PR to check if a trusted
    user has flagged this user as trusted using the `/bot trust user X`
    command.

    Parameters
    ----------
    pr_id : int
        The number of the PR.

    user : str
        The username of the user who may be untrustworthy.

    Results
    -------
    bool : True if trustworthy, false otherwise.
    """
    trusted_comments = [c for c in get_previous_pr_comments(pr_id) if c.author in trusted_reviewers]
    for c in trusted_comments:
        words = c.body.strip().split()
        if words == ['/bot', 'trust', 'user', user]:
            return True

    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Call the function to activate the bot')
    parser.add_argument('gitEvent', type=str,
                        help='File containing the json description of the triggering event')
    parser.add_argument('output', type=str,
                        help='File where the variables should be saved')
    parser.add_argument('run_id', type=int,
                        help='The id of the runner (used to identify the action page)')
    parser.add_argument('cleanup_trigger', type=str, nargs='?', default='',
                        help='A key to be passed when the bot is used to clean up after tests have run [request_review_status,update_test_information]')

    args = parser.parse_args()

    # Parse event payload from $GITHUB_EVENT_PATH variable
    # (documented here : https://docs.github.com/en/actions/learn-github-actions/variables#default-environment-variables)
    # The contents of this json file depend on the triggering event and are
    # described here :  https://docs.github.com/en/webhooks-and-events/webhooks/webhook-events-and-payloads
    with open(args.gitEvent, encoding="utf-8") as event_file:
        event = json.load(event_file)

    # Save run number into dictionary with event information
    event['run_number'] = args.run_id

    cleanup_trigger = args.cleanup_trigger

    # Initialise outputs
    outputs = {'run_linux': False,
               'run_windows': False,
               'run_macosx': False,
               'run_coverage': False,
               'run_docs': False,
               'run_pylint': False,
               'run_lint': False,
               'run_spelling': False,
               'run_pickle': False,
               'run_editable_pickle': False,
               'run_pickle_wheel': False,
               'run_linux_anaconda': False,
               'run_windows_anaconda': False,
               'cleanup_trigger': '',
               'python_version': '',
               'BASE': '',
               'REF': '',
               'SHA': ''}

    if cleanup_trigger == 'request_review_status':
        # Collect id from an issue_comment event with a created action
        # or from a pull_request event with a synchronize action
        if 'number' in event:
            pr_id = event['number']
        else:
            pr_id = event['issue']['number']
        update_test_information(pr_id, event)
        result = mark_as_ready(pr_id)
        with open(args.output, encoding="utf-8", mode='a') as out_file:
            print(f"global_state={result}", file=out_file)
        sys.exit()

    elif cleanup_trigger == 'update_test_information':
        # If reporting after run

        # Collect id from a comment event
        pr_id = event['issue']['number']

        result = update_test_information(pr_id, event)
        with open(args.output, encoding="utf-8", mode='a') as out_file:
            print(f"global_state={result}", file=out_file)
        sys.exit()

    elif 'comment' in event and 'pull_request' in event['issue'] and event['comment']['body'].startswith('/bot'):
        # If bot called explicitly (comment event)

        # Collect id from an issue_comment event with a created action
        pr_id = event['issue']['number']

        trusted_user = event['comment']['author_association'] in ('COLLABORATOR', 'CONTRIBUTOR', 'MEMBER', 'OWNER')
        print("Trust level : ", event['comment']['author_association'])
        if not trusted_user:
            trusted_user = flagged_as_trusted(pr_id, event['comment']['user']['login'])

        comment = event['comment']['body']
        command = comment.split('/bot')[1].strip()
        command_words = command.split()

        if command_words[0] == 'run':
            if trusted_user:
                if run_tests(pr_id, command_words[1:], outputs, event):
                    outputs['cleanup_trigger'] = 'update_test_information'
            else:
                leave_comment(pr_id, message_from_file('untrusted_user.txt'))

        elif command_words[0] == 'try':
            if trusted_user:
                outputs['python_version'] = command_words[1]
                if run_tests(pr_id, command_words[2:], outputs, event):
                    outputs['cleanup_trigger'] = 'update_test_information'
            else:
                leave_comment(pr_id, message_from_file('untrusted_user.txt'))

        elif command == 'mark as ready':
            if trusted_user:
                start_review_check(pr_id, event, outputs)
            else:
                leave_comment(pr_id, message_from_file('untrusted_user.txt'))

        elif command == 'show tests':
            leave_comment(pr_id, message_from_file('show_tests.txt'))

        elif command_words[:2] == ['trust', 'user'] and len(command_words)==3:
            user = command_words[2]
            leave_comment(pr_id, message_from_file('trusting_user.txt').format(user=user))
            pr_author = get_status_json(pr_id, 'author')
            draft = get_status_json(pr_id, 'isDraft')
            if not draft and user == pr_author:
                outputs['cleanup_trigger'] = 'request_review_status'
                run_tests(pr_id, ['pr_tests'], outputs, event)

        else:
            leave_comment(pr_id, message_from_file('bot_commands.txt'))

    elif event['action'] == 'opened':
        # If new PR (opened trigger)

        # Collect id from a pull request event with an opened action
        pr_id = event['number']

        # Check whether user is new and/or trusted
        trusted_user = event['pull_request']['author_association'] in ('COLLABORATOR', 'CONTRIBUTOR', 'MEMBER', 'OWNER')
        if trusted_user:
            prs = check_previous_contributions(event['repository']['full_name'], event['pull_request']['user']['login'])
            new_user = (len(prs) == 0)
        else:
            new_user = True

        # Choose appropriate message to welcome author
        file = 'welcome_newcomer.txt' if new_user else 'welcome_friend.txt'

        leave_comment(pr_id, message_from_file(file) + message_from_file('checklist.txt'))

        # Ensure PR is draft
        if not event['pull_request']['draft']:
            set_draft(pr_id)

        # If unknown user ask for trust approval
        if not trusted_user:
            leave_comment(pr_id, ", ".join(senior_reviewer)+", please can you check if I can trust this user. If you are happy, let me know with `/bot trust user`")

    elif event['action'] == 'converted_to_draft':
        # Collect id from a pull_request event with a converted_to_draft action
        pr_id = event['number']

        remove_labels(pr_id, ['Ready_to_merge', 'Ready_for_review', 'needs_initial_review'])

    elif 'review' in event:
        pr_id = event['pull_request']['number']
        state = event['review']['state']
        if state == 'approved':
            labels = get_status_json(pr_id, 'labels')
            remove_labels(pr_id, ['Ready_to_merge', 'Ready_for_review', 'needs_initial_review'])
            set_review_stage(pr_id)
        elif state == 'changes_requested':
            labels = get_status_json(pr_id, 'labels')
            remove_labels(pr_id, ['Ready_to_merge', 'Ready_for_review', 'needs_initial_review'])
            set_draft(pr_id)
            author = get_status_json(pr_id, 'author')['login']
            reviewer = event['review']['user']['login']
            leave_comment(pr_id, message_from_file('set_draft_changes.txt').format(author=author, reviewer=reviewer))

    elif 'pull_request' in event and not event['pull_request']['draft']:
        # If PR is ready for review

        # Collect id from a pull_request event
        pr_id = event['number']
        trusted_user = event['pull_request']['author_association'] in ('COLLABORATOR', 'CONTRIBUTOR', 'MEMBER', 'OWNER')
        if not trusted_user:
            trusted_user = flagged_as_trusted(pr_id, event['comment']['user']['login'])

        if trusted_user:
            start_review_check(pr_id, event, outputs)
        else:
            leave_comment(pr_id, message_from_file('untrusted_user.txt'))

    else:
        pr_id = None

    if pr_id is not None:
        status = get_status_json(pr_id, 'baseRefName,headRefOid')
        outputs['BASE'] = status['baseRefName']
        outputs['REF'] = f'refs/pull/{pr_id}/merge'
        outputs['SHA'] = status['headRefOid']

    # Print the output to the file described by $GITHUB_OUTPUT to create outputs
    # as described here : https://docs.github.com/en/actions/using-workflows/workflow-commands-for-github-actions#setting-an-output-parameter
    with open(args.output, encoding="utf-8", mode='a') as out_file:
        for o,v in outputs.items():
            print(f"{o}={v}", file=out_file)
            print(f"{o}={v}")
