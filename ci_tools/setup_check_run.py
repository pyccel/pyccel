import os
from bot_tools.bot_funcs import Bot

bot = Bot(check_run_id = os.environ["GITHUB_RUN_ID"], commit = os.environ["GITHUB_REF"])
if os.environ["GITHUB_RUN_ID"]=="":
    bot.create_in_progress_check_run()
else:
    print(os.environ["GITHUB_RUN_ID"])
    bot.post_in_progress()
