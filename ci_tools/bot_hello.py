import json
import os
from bot_tools.bot_funcs import Bot

senior_reviewer = ['yguclu', 'EmilyBourne', 'bauom']

# Parse event payload from $GITHUB_EVENT_PATH variable
# (documented here : https://docs.github.com/en/actions/learn-github-actions/variables#default-environment-variables)
# The contents of this json file depend on the triggering event and are
# described here :  https://docs.github.com/en/webhooks-and-events/webhooks/webhook-events-and-payloads
with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as event_file:
    event = json.load(event_file)

# Collect id from a pull request event with an opened action
pr_id = event['number']

bot = Bot(pr_id = pr_id)

user = event['pull_request']['user']['login']

# Check whether user is new and/or trusted
trusted_user = bot.is_user_trusted(user)
print("Current user trust level is : ", event['pull_request']['author_association'])
if trusted_user:
    merged_prs = bot.GAI.get_prs('all')
    has_merged_pr = any(pr for pr in merged_prs if pr['user']['login'] == user and pr['merged_at'])
    new_user = has_merged_pr
else:
    new_user = True

# Choose appropriate message to welcome author
file = 'welcome_newcomer.txt' if new_user else 'welcome_friend.txt'

leave_comment(pr_id, message_from_file(file) + message_from_file('checklist.txt'))

# Ensure PR is draft
if not event['pull_request']['draft']:
    bot.mark_as_draft(pr_id)

# If unknown user ask for trust approval
if not trusted_user:
    bot.GAI.leave_comment(pr_id, ", ".join(f'@{r}' for r in senior_reviewer)+", please can you check if I can trust this user. If you are happy, let me know with `/bot trust user`")

