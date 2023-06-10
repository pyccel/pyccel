import argparse
import json
import os
from bot_tools.bot_funcs import Bot

pr_test_keys = ['linux', 'windows', 'macosx', 'coverage', 'doc_coverage', 'pylint',
                'pyccel_lint', 'spelling']

def get_unique_test_list(keys):
    tests = set(command_words[1:])
    t = tests.discard('pr_tests')
    if t:
        tests.update(pr_test_keys)
    if 'coverage' in tests:
        tests.add('linux')
    return tests

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

bot = Bot(pr_id)

if command_words[:2] == ['show', 'tests']:
    bot.show_tests()

elif command_words[0] == 'run':
    if bot.is_user_trusted(event['comment']['user']['login']):
        bot.run_tests(get_unique_test_list(command_words[1:]))
    else:
        bot.warn_untrusted()

elif command_words[0] == 'try':
    if bot.is_user_trusted(event['comment']['user']['login']):
        bot.run_tests(get_unique_test_list(command_words[1:]), command_words[1])
    else:
        bot.warn_untrusted()

elif command_words[:3] == ['mark', 'as', 'ready']:
    if bot.is_user_trusted(event['comment']['user']['login']):
        bot.request_mark_as_ready()
        bot.run_tests(pr_test_keys)
    else:
        bot.warn_untrusted()

elif command_words[:2] == ['trust', 'user'] and len(command_words)==3 and event['comment']['user']['login'] in Bot.trust_givers:

    bot.indicate_trust(command_words[2])

else:
    bot.show_commands()
