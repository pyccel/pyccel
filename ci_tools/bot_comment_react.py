""" Script run as a reaction to a comment left on a pull request.
"""
import json
import os
from bot_tools.bot_funcs import Bot, pr_test_keys

def get_unique_test_list(keys):
    """
    Get a list of unique tests which should be run.

    Get a list of all the tests which should be run. The treatments to
    obtain this list remove duplicate tests. It also expands the `pr_tests`
    key to all tests necessary for a pull request. Finally if coverage is
    requested, it additionally requests linux.

    Parameters
    ----------
    keys : list of str
        The list of tests requested by the user.

    Returns
    -------
    set
        A set of tests to run.
    """
    tests = set(keys)
    if 'pr_tests' in tests:
        tests.update(pr_test_keys)
    tests.discard('pr_tests')
    if 'coverage' in tests:
        tests.add('linux')
        # Ensure coverage is last in case dependencies are ready
        tests.discard('coverage')
        result = list(tests)
        result.append('coverage')
    else:
        result = list(tests)
    return result

if __name__ == '__main__':
    # Parse event payload from $GITHUB_EVENT_PATH variable
    # (documented here : https://docs.github.com/en/actions/learn-github-actions/variables#default-environment-variables)
    # The contents of this json file depend on the triggering event and are
    # described here :  https://docs.github.com/en/webhooks-and-events/webhooks/webhook-events-and-payloads
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as event_file:
        event = json.load(event_file)

    # If bot called explicitly (comment event)

    # Collect id from an issue_comment event with a created action
    pr_id = event['issue']['number']

    comment = event['comment']['body']
    command = comment.split('/bot')[1].strip()
    command_words = command.split()

    bot = Bot(pr_id = pr_id)

    if command_words[:2] == ['show', 'tests']:
        bot.show_tests()

    elif command_words[0] == 'checklist':
        bot.fill_checklist(event['comment']['url'], event['comment']['user']['login'])

    elif command_words[0] == 'run':
        if bot.is_user_trusted(event['comment']['user']['login']):
            bot.run_tests(get_unique_test_list(command_words[1:]))
        else:
            bot.warn_untrusted()

    elif command_words[0] == 'try':
        if bot.is_user_trusted(event['comment']['user']['login']):
            print(command_words, get_unique_test_list(command_words[2:]), command_words[1])
            bot.run_tests(get_unique_test_list(command_words[2:]), command_words[1])
        else:
            bot.warn_untrusted()

    elif command_words[:3] == ['mark', 'as', 'ready']:
        if bot.is_user_trusted(event['comment']['user']['login']):
            bot.request_mark_as_ready()
        else:
            bot.warn_untrusted()

    elif command_words[:2] == ['trust', 'user'] and len(command_words)==3 and event['comment']['user']['login'] in Bot.trust_givers:

        bot.indicate_trust(command_words[2])

    else:
        bot.show_commands()
