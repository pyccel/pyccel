""" Script to determine which stage of the review is in progress and add appropriate labels
"""
from datetime import datetime
import json
import namedtuple
import shutil
import subprocess
import sys

github_cli = shutil.which('gh')

ReviewComment = namedtuple('ReviewComment', ['state', 'date'])

possible_comments = {'REVIEW_FOR_DRAFT', 'This PR is asking for a review but it is still in draft. Please either remove the Draft status or remove the labels.',
                     'MULTIPLE_STAGES', 'This PR has too many review flags. Please check the [review process docs](https://github.com/pyccel/pyccel/blob/master/developer_docs/review_process.md) to find the correct flag.',
                     'NOT_PASSING',"Unfortunately this PR does not yet pass the required tests. I'm going to flag it as a Draft for now. Please remove the Draft status when you think it is ready for review.",
                     'FIRST_REVIEW_OK': "Congratulations, your first review seems to have been successful. I'm moving this on to the next stage of the review process.",
                     'CONGRATULATIONS': "Congratulations, a senior developer thinks your PR looks great! :tada:\nThis PR is nearly ready to merge I will move it to the final stage for merging.",
                     'REVIEWS_GONE': "There don't seem to be any successful reviews here. I'm going to move this PR back to the start of the review cycle.",
                     'SENIOR_REVIEWS_CHANGE': "Sorry, it looks like a senior developer has found an issue with this PR. I'm going to move it back to the \"Ready_for_review\" stage to make sure this problem is fully scrutinised."
                     }

status_labels = {'needs_initial_review':1, 'Ready_for_review':2, 'Ready_to_merge':3}

senior_reviewers = ('yguclu', 'EmilyBourne', 'saidctb', 'bauom')

comment_lookup = dict([reversed(i) for i in possible_comments.items()])

def get_pr_number():
    """
    Check if this branch has exactly 1 related PR.

    Use GitHub's command-line tool to get the PR number for
    all PRs related to this branch. Output a message to clarify
    which PR is considered relevant.

    Results
    -------
    bool : Indicates whether this branch has exactly 1 related PR.
    """
    cmds = [github_cli, 'pr', 'status', '--json', 'number']

    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    result, err = p.communicate()

    output = json.loads(result)

    if 'currentBranch' not in output:
        return []
    else:
        PRs = output['currentBranch']['number']
        if isinstance(PRs, list):
            return PRs
        else:
            return PRs



def get_labels():
    """
    Get the labels associated with the PR.

    Use GitHub's command-line tool to get all labels for
    all PRs related to this branch.

    Results
    -------
    list : A list of the names of all the labels.
    """
    cmds = [github_cli, 'pr', 'status', '--json', 'labels']

    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    result, err = p.communicate()

    label_json = json.loads(result)['currentBranch']['labels']
    current_labels = [l['name'] for l in label_json]
    return current_labels



def is_draft():
    """
    Get the draft status of the PR.

    Use GitHub's command-line tool to get the draft status for
    the first PR related to this branch.

    Results
    -------
    bool : The draft status of the PR.
    """
    cmds = [github_cli, 'pr', 'status', '--json', 'isDraft']

    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    result, err = p.communicate()

    return json.loads(result)['currentBranch']['isDraft']



def get_review_status():
    """
    Get the reviews left on the PR.

    Get a dictionary describing the current state of the reviews
    on the PR. 

    Results
    -------
    dict : Keys are authors of reviews, values are the state of their review.
    """
    cmds = [github_cli, 'pr', 'status', '--json', 'reviews']

    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    result, err = p.communicate()

    reviews = json.loads(result)['currentBranch']['reviews']

    cmds = [github_cli, 'pr', 'status', '--json', 'reviewRequests']

    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    result, err = p.communicate()

    requests = json.loads(result)['currentBranch']['reviewRequests']
    requested_authors = [r["login"] for r in requests]

    review_status = {}
    for r in reviews:
        author = r['author']['login']
        date = datetime.fromisoformat(r['submittedAt'].strip('Z'))
        state = r['state']
        if author not in review_status:
            review_status[author] = ReviewComment(state, date)
        elif state != 'COMMENTED' and review_status[author].date < date:
            review_status[author] = ReviewComment(state, date)
    for a in review_status:
        if a in requested_authors:
            review_status[a] = ReviewComment('REVIEW_REQUESTED', review_status[a].date)
    for a in requests:
        if a not in review_status:
            review_status[a] = ReviewComment('UNRESPONSIVE', None)
    return review_status, requested_authors


def check_previous_comments():
    cmds = [github_cli, 'pr', 'status', '--json', 'comments']

    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    result, err = p.communicate()

    previous_comments = json.loads(result)['currentBranch']['comments']

    my_comments = [ReviewComment(c["body"], datetime.fromisoformat(c['createdAt'].strip('Z')))
                        for c in previous_comments if ['author']['login'] == 'github-actions']

    if len(my_comments) == 0:
        return [], '', None
    else:
        last_messages = {}
        final_message = comment_lookup.get(my_comments[0].state, None)
        final_date = my_comments[0].date

        for c in my_comments:
            key = comment_lookup.get(c.state, None)
            assert key is not None
            if key not in last_message:
                last_message[key] = c.date
            elif last_message[key] < c.date:
                last_message[key] = c.date

            if final_date < c.date:
                final_message = c.state
                final_date = c.date

        return last_messages, final_message, final_date

def leave_comment(number, comment_key):
    """
    Leave a comment on the PR.

    Use GitHub's command-line interface to leave a comment
    on the request PR related to this branch.

    Parameters
    ----------
    number : int
        The number of the PR.

    comment : str
        The key to a comment which should be left on the PR (as defined in possible_comments).
    """
    previous_comments, last_comment, last_date = check_previous_comments()

    comment = possible_comments[comment_key]

    if last_comment != comment:
        cmds = [github_cli, 'pr', 'comment', str(number), '-b', f'"{comment}"']

        print("Commenting : ", cmds)

        #p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
        #result, err = p.communicate()

def set_draft(number):
    """
    Set PR to draft and remove review labels.

    Use GitHub's command-line interface to set the PR to a draft and
    remove the labels indicating the review stage.

    Parameters
    ----------
    number : int
        The number of the PR.
    """
    print("Draft")
    return
    cmds = [github_cli, 'pr', 'ready', '--undo']

    p = subprocess.Popen(cmds)
    p.communicate()

    cmds = [github_cli, 'pr', 'edit']
    for lab in status_labels:
        cmds += ['--remove-label', lab]

    p = subprocess.Popen(cmds)
    p.communicate()


def check_passing():
    """
    Check if tests are passing for this PR.

    Results
    -------
    bool : Indicates if tests are passed
    """

    # Wait till tests have finished
    cmds = [github_cli, 'pr', 'checks', '--required', '--watch']

    p = subprocess.Popen(cmds)
    p.communicate()


    # Collect results
    cmds = [github_cli, 'pr', 'status', '--json', 'statusCheckRollup']

    p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    result, err = p.communicate()

    checks = json.loads(result)
    passing = all(c['conclusion'] == 'SUCCESS' for c in checks if c['name'] not in ('CoverageChecker', 'Check labels', 'Welcome'))

    return passing



def set_status(current_status, new_status, hanging_authors):
    """
    Set the PR to the requested stage of the review process.

    Set the necessary labels and request any additional reviewers appropriate
    for the new stage.

    Parameters
    ----------
    current_status : str
        The current stage of the review process.

    new_status : str
        The stage of the review process being set.

    hanging_authors : list
        A list of authors whose review has been requested but who have
        not yet reviewed.
    """
    if current_status == new_status:
        return

    cmd = [github_cli, 'pr', 'edit', '--add-label', new_status, '--remove-label', current_status]

    if current_status == '':
        cmd = cmd[:-2]
        leave_comment(pr_id, 'READY_FOR_REVIEW')
        cmd.extend(['--add-reviewer', 'pyccel/pyccel-dev'])
        
    elif current_status == 'needs_initial_review':
        if new_status == 'Ready_for_review':
            leave_comment(pr_id, 'FIRST_REVIEW_OK')
            cmd.extend(['--add-reviewer', 'EmilyBourne'])

        elif new_status == 'Ready_to_merge':
            leave_comment(pr_id, 'CONGRATULATIONS')
            cmd.extend(['--add-reviewer', 'yguclu'])

    elif current_status == 'Ready_for_review':
        if new_status == 'needs_initial_review':
            leave_comment(pr_id, 'REVIEWS_GONE')
            cmd.extend(['--add-reviewer', 'pyccel/pyccel-dev'])

        elif new_status == 'Ready_to_merge':
            leave_comment(pr_id, 'CONGRATULATIONS')
            cmd.extend(['--add-reviewer', 'yguclu'])

    elif current_status == 'Ready_to_merge':
        if new_status == 'Ready_for_review':
            leave_comment(pr_id, 'SENIOR_REVIEWS_CHANGE')

        elif new_status == 'Ready_for_review':
            leave_comment(pr_id, 'REVIEWS_GONE')
            cmd.extend(['--add-reviewer', 'pyccel/pyccel-dev'])

    if new_status != 'needs_initial_review':
        requested_authors_to_keep = senior_reviewers if new_status == 'Ready_for_review' else ('yguclu',)

        for a in hanging_authors:
            if a not in requested_authors_to_keep:
                cmd.extend(['--remove-reviewer', a])

    p = subprocess.Popen(cmd)
    p.communicate()

if __name__ == '__main__':
    pr_id = get_pr_number()
    if isinstance(pr_id, list):
        if pr_id:
            print("Multiple PRs open for this branch : ", ", ".join(PRs))
        else:
            print("No PR open for this branch")
        exit(1)

    print("Examining PR", pr_id)

    draft = is_draft()
    labels = get_labels()
    flagged_status = set(labels).intersection(status_labels.keys())

    print(draft, labels)
    print(draft, flagged_status)

    if draft:
        if flagged_status:
            leave_comment(pr_id, 'REVIEW_FOR_DRAFT')
        sys.exit(0)

    # Not draft

    if len(flagged_status) > 1:
        leave_comment(pr_id, 'MULTIPLE_STAGES')
        sys.exit(0)

    passing = check_passing()

    if not passing:
        leave_comment(pr_id, 'NOT_PASSING')
        set_draft()
        sys.exit(0)

    # Passing

    reviews, hanging_authors = get_review_status()
    approved = any(r.state == 'APPROVED' for r in reviews.values())
    senior_approved = any(r.state == 'APPROVED' for a,r in reviews.items() if a in senior_reviewers)
    requested_changes = [a for a,r in reviews.items() if r == 'CHANGES_REQUESTED']

    if approved and senior_approved and not any(r in senior_reviewers for r in requested_changes):
        predicted_status = 'Ready_to_merge'
    elif approved and not any(r not in senior_reviewers for r in requested_changes)
        predicted_status = 'Ready_for_review'
    else:
        predicted_status = 'needs_initial_review'

    if predicted_status == 'needs_initial_review' and approved:
        if 

    if flagged_status:
        flagged_status = flagged_status.pop()
        if flagged_status == 'Ready_to_merge':
            if not senior_approved:
                status = 'Ready_for_review'
            elif not approved:
                status = 'needs_initial_review'
            else:
                status = 'Ready_to_merge'
        elif flagged_status == 'Ready_for_review':
            if not approved:
                status = 'needs_initial_review'
            else:
                status = 'Ready_for_review'
        else:
            status = 'needs_initial_review'
    else:
        status = ''

    if status_labels[predicted_status] > status_labels.get(status, 0):
        # Move to next review stage
        predicted_status = status

    set_status(flagged_status, status)
