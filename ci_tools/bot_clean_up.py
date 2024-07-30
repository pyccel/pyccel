""" Script run after a check run to trigger coverage tests if necessary, and change the draft status if necessary.
"""
import json
import os
import sys
from bot_tools.bot_funcs import Bot, test_dependencies, pr_test_keys as bot_pr_test_keys, default_python_versions

pr_test_keys = [(t, default_python_versions[t]) for t in bot_pr_test_keys] + ['Codacy']

def get_name_from_key(name_key):
    """
    Get the name from a name key by dropping the information about the Python version.

    Get the name from a name key by dropping the information about the Python version.

    Parameters
    ----------
    name_key : str | tuple
        The key used to uniquely identify the run configuration.

    Returns
    -------
    str
        A string indicating the run tool (but not the Python version).
    """
    if isinstance(name_key, tuple):
        # Ignore Python version
        return name_key[0]
    return name_key

if __name__ == '__main__':
    # Parse event payload from $GITHUB_EVENT_PATH variable
    # (documented here : https://docs.github.com/en/actions/learn-github-actions/variables#default-environment-variables)
    # The contents of this json file depend on the triggering event and are
    # described here :  https://docs.github.com/en/webhooks-and-events/webhooks/webhook-events-and-payloads
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as event_file:
        event = json.load(event_file)

    full_name = event['check_run']['name']

    print(event)

    print(event['check_run']['pull_requests'])
    try:
        pr_id = next(p['number'] for p in event['check_run']['pull_requests'])
    except StopIteration:
        pr_id = 0
    bot = Bot(pr_id = pr_id, commit = event['check_run']['head_sha'])
    try:
        pr_id = bot.get_pr_id()
    except StopIteration:
        pr_id = 0

    name = get_name_from_key(bot.get_name_key(full_name))

    runs = bot.get_check_runs()

    print("Runs: ", runs)

    all_run_names = [bot.get_name_key(r['name']) for r in runs]
    successful_runs = [n for n,r in zip(all_run_names, runs) if r['conclusion'] == "success"]
    completed_runs = [n for n,r in zip(all_run_names, runs) if r['status'] == "completed"]
    queued_runs = [r for r in runs if r['status'] == "queued"]

    print("Successful:", successful_runs)
    print("Completed:", completed_runs)

    for q in queued_runs:
        q_name = q['name']
        q_key = bot.get_name_key(q_name)
        if not isinstance(q_key, tuple):
            # Not a Pyccel triggered test
            continue
        deps = test_dependencies.get(q_key[0], ())
        if name in deps:
            if all(d in (get_name_from_key(r) for r in successful_runs) for d in deps):
                q_name, python_version = q_key
                workflow_ids = None
                if q_name == 'coverage':
                    workflow_ids = [int(r['details_url'].split('/')[-1]) for r in runs if r['conclusion'] == "success" and '(' in r['name']]
                bot.run_test(q_name, python_version, q["id"], workflow_ids)

    if pr_id != 0:
        draft = bot.is_pr_draft()

        if not draft:

            events = bot.GAI.get_events(pr_id)

            shas = [e.get('sha', None) for e in events]
            if events[-1].get('event', None) == 'closed':
                sys.exit(0)

            print(shas)
            print([e.get('event', None) for e in events])
            page = 1
            start_idx = -1
            while start_idx == -1:
                try:
                    start_idx = next(i for i,s in enumerate(shas) if s == event['check_run']['head_sha'])
                except StopIteration:
                    start_idx = -1
                    page += 1
                    new_events = bot.GAI.get_events(pr_id, page)
                    events.extend(new_events)
                    shas.extend([e.get('sha', None) for e in new_events])
                    print(shas)
            try:
                end_idx = next(i for i,s in enumerate(shas[start_idx+1:], start_idx+1) if s is not None)
            except StopIteration:
                end_idx = len(shas)

            relevant_events = events[:end_idx]

            event_types = [e['event'] for e in events]

            relevant_ready_events = [e for e in event_types[:end_idx] if e in ('ready_for_review', 'convert_to_draft')]
            later_ready_events = [e for e in event_types[end_idx:] if e in ('ready_for_review', 'convert_to_draft')]

            was_examined = relevant_ready_events and relevant_ready_events[-1] == 'ready_for_review'
            result_ignored = bool(later_ready_events)

            print(was_examined, result_ignored)

            if was_examined and not result_ignored:
                print(completed_runs, pr_test_keys, successful_runs)
                print(all(k in completed_runs for k in pr_test_keys),
                     all(k in successful_runs for k in pr_test_keys))
                if event['check_run']['conclusion'] == 'failure':
                    bot.draft_due_to_failure()
                elif event['check_run']['conclusion'] not in ('success', 'skipped'):
                    bot.mark_as_draft()
                elif all(k in completed_runs for k in pr_test_keys) and \
                     all(k in successful_runs for k in pr_test_keys):
                    bot.mark_as_ready(following_review = False)
