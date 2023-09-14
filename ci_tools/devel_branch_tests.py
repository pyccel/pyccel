""" A script to run the `devel` branch tests.
"""
import os
from bot_tools.bot_funcs import Bot

if __name__ == '__main__':
    bot = Bot(pr_id = 0, commit = os.environ['GITHUB_REF'])
    for python_version in ('3.8', '3.9', '3.10', '3.11'):
        bot.run_tests(['linux'], python_version, force_run = True)
    bot.run_tests(['windows'], '3.8', force_run = True)
    bot.run_tests(['macosx'], '3.9', force_run = True)
    bot.run_tests(['pickle'], '3.8', force_run = True)
    bot.run_tests(['editable_pickle'], '3.8', force_run = True)
    bot.run_tests(['pickle_wheel'], '3.8', force_run = True)
    bot.run_tests(['anaconda_linux'], '3.10', force_run = True)
    bot.run_tests(['anaconda_windows'], '3.10', force_run = True)
