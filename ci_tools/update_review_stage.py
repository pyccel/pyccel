""" Script to determine which stage of the review is in progress and add appropriate labels
"""
import datetime
import subprocess
import sys

from git_evaluation_tools import github_cli, get_pr_number, get_labels
from git_evaluation_tools import is_draft, get_review_status, check_passing, leave_comment
from git_evaluation_tools import remove_labels, set_draft

possible_comments = {'REVIEW_FOR_DRAFT': 'This PR is asking for a review but it is still in draft. Please either remove the Draft status or remove the labels.',
                     'MULTIPLE_STAGES': 'This PR has too many review flags. Please check the [review process docs](https://github.com/pyccel/pyccel/blob/master/developer_docs/review_process.md) to find the correct flag.',
                     'NOT_PASSING':"Unfortunately this PR does not yet pass the required tests. I'm going to flag it as a Draft for now. Please remove the Draft status when you think it is ready for review.",
                     'READY_FOR_REVIEW':'This PR looks like it is ready for review. I will add the appropriate labels for you.',
                     'FIRST_REVIEW_OK': "Congratulations, your first review seems to have been successful. I'm moving this on to the next stage of the review process.",
                     'CONGRATULATIONS': "Congratulations, a senior developer thinks your PR looks great! :tada:\nThis PR is nearly ready to merge I will move it to the final stage for merging.",
                     'REVIEWS_GONE': "There don't seem to be any successful reviews here. I'm going to move this PR back to the start of the review cycle.",
                     'SENIOR_REVIEWS_CHANGE': "Sorry, it looks like a senior developer has found an issue with this PR. I'm going to move it back to the \"Ready_for_review\" stage to make sure this problem is fully scrutinised.",
                     'REVIEW_NUDGE':"Hi {changes}, it looks like {approved} has approved this PR. Do you agree with them? We need your review to move on to the next stage of the review process."
                     }

status_labels = {'needs_initial_review':1, 'Ready_for_review':2, 'Ready_to_merge':3}

senior_reviewers = ('yguclu', 'EmilyBourne', 'saidctb')



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
        leave_comment(pr_id, possible_comments['READY_FOR_REVIEW'], True)
        if new_status == 'needs_initial_review':
            cmd.extend(['--add-reviewer', 'pyccel/pyccel-dev'])
        elif new_status == 'Ready_for_review':
            cmd.extend(['--add-reviewer', 'EmilyBourne'])
        elif new_status == 'Ready_to_merge':
            cmd.extend(['--add-reviewer', 'yguclu'])

    elif current_status == 'needs_initial_review':
        if new_status == 'Ready_for_review':
            leave_comment(pr_id, possible_comments['FIRST_REVIEW_OK'], True)
            cmd.extend(['--add-reviewer', 'EmilyBourne'])

        elif new_status == 'Ready_to_merge':
            leave_comment(pr_id, possible_comments['CONGRATULATIONS'], True)
            cmd.extend(['--add-reviewer', 'yguclu'])

    elif current_status == 'Ready_for_review':
        if new_status == 'needs_initial_review':
            leave_comment(pr_id, possible_comments['REVIEWS_GONE'], True)
            cmd.extend(['--add-reviewer', 'pyccel/pyccel-dev'])

        elif new_status == 'Ready_to_merge':
            leave_comment(pr_id, possible_comments['CONGRATULATIONS'], True)
            cmd.extend(['--add-reviewer', 'yguclu'])

    elif current_status == 'Ready_to_merge':
        if new_status == 'Ready_for_review':
            leave_comment(pr_id, possible_comments['SENIOR_REVIEWS_CHANGE'], True)

        elif new_status == 'Ready_for_review':
            leave_comment(pr_id, possible_comments['REVIEWS_GONE'], True)
            cmd.extend(['--add-reviewer', 'pyccel/pyccel-dev'])

    if new_status != 'needs_initial_review':
        requested_authors_to_keep = senior_reviewers if new_status == 'Ready_for_review' else ('yguclu',)

        for a in hanging_authors:
            if a not in requested_authors_to_keep:
                cmd.extend(['--remove-reviewer', a])

    with subprocess.Popen(cmd) as p:
        p.communicate()

if __name__ == '__main__':
    pr_id = get_pr_number()
    if isinstance(pr_id, list):
        if pr_id:
            print("Multiple PRs open for this branch : ", ", ".join(pr_id))
        else:
            print("No PR open for this branch")
        sys.exit(1)

    print("Examining PR", pr_id)

    draft = is_draft()
    labels = get_labels()
    flagged_status = set(labels).intersection(status_labels.keys())

    if draft:
        if flagged_status:
            leave_comment(pr_id, possible_comments['REVIEW_FOR_DRAFT'], False)
        sys.exit(0)

    print("Not a draft")
    # Not draft

    if len(flagged_status) > 1:
        leave_comment(pr_id, possible_comments['MULTIPLE_STAGES'], False)
        sys.exit(0)

    print("Flags make sense")
    passing = check_passing()

    if not passing:
        leave_comment(pr_id, possible_comments['NOT_PASSING'], True)
        set_draft(pr_id)
        remove_labels(pr_id, status_labels)
        sys.exit(0)

    print("Tests passing")
    # Passing

    reviews, hanging_authors = get_review_status()
    approval = {a:r for a,r in reviews.items() if r.state == 'APPROVED'}
    requested_changes = {a:r for a,r in reviews.items() if r == 'CHANGES_REQUESTED'}

    approved = len(approval)>0
    senior_approved = any(a in senior_reviewers for a in approval)

    if approved and senior_approved and not any(r in senior_reviewers for r in requested_changes):
        predicted_status = 'Ready_to_merge'
    elif approved and not any(r not in senior_reviewers for r in requested_changes):
        predicted_status = 'Ready_for_review'
    else:
        predicted_status = 'needs_initial_review'

    print(f"I think the status should be {predicted_status}")

    if predicted_status == 'needs_initial_review' and approved:
        last_approved = max(r.date for r in approval.values())
        if all((datetime.utcnow() - last_approved).days > 2):
            approvers = ', '.join(f'@{a}' for a in approval)
            requesters = ', '.join(f'@{a}' for a in requested_changes)
            leave_comment(pr_id, possible_comments['REVIEW_NUDGE'].format(changes = requesters, approved = approvers), False)

    if flagged_status:
        print(f"The current status is {flagged_status}")
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
        print(f"The current status is {status}")
    else:
        flagged_status = None
        status = ''

    if status_labels[predicted_status] > status_labels.get(status, 0):
        # Move to next review stage
        status = predicted_status

    print(f"Changing status from {flagged_status} to {status}")

    set_status(flagged_status, status, hanging_authors)
