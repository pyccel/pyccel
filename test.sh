INSTALLATION_TOKEN="$(python3 ci_tools/bot_tools/generate_jwt.py /home/emily/Downloads/testpyccelbot.2023-05-23.private-key.pem)"

echo ${INSTALLATION_TOKEN}

## Get installation token
#curl --request POST \
#--url "https://api.github.com/app/installations/37820767/access_tokens" \
#--header "Accept: application/vnd.github+json" \
#--header "Authorization: Bearer ${TOKEN}" \
#--header "X-GitHub-Api-Version: 2022-11-28"

#["token"]

# Check runs (https://docs.github.com/en/rest/checks/runs?apiVersion=2022-11-28)
#curl -L \
#  -H "Accept: application/vnd.github+json" \
#  -H "Authorization: Bearer ${INSTALLATION_TOKEN}"\
#  -H "X-GitHub-Api-Version: 2022-11-28" \
#  https://api.github.com/repos/EmilyBourne/pyccel/commits/b54121ae9bb8d190e093876578c8f5eac1d9f669/check-runs

# Create run
#curl -L \
#  -X POST \
#  -H "Accept: application/vnd.github+json" \
#  -H "Authorization: Bearer ${INSTALLATION_TOKEN}"\
#  -H "X-GitHub-Api-Version: 2022-11-28" \
#  --url https://api.github.com/repos/EmilyBourne/pyccel/check-runs \
#  -d '{"name":"mighty_readme","head_sha":"c9fa506c4b047fa28f0edf48d4385d8d6c2c17f5","status":"in_progress","external_id":"42"}'

# ["check_runs"][0]["id"]

# Update run
#curl -L \
#  -X PATCH \
#  -H "Accept: application/vnd.github+json" \
#  -H "Authorization: Bearer ${INSTALLATION_TOKEN}"\
#  -H "X-GitHub-Api-Version: 2022-11-28" \
#  https://api.github.com/repos/EmilyBourne/pyccel/check-runs/13989855986 \
#  -d '{"status":"completed","conclusion":"success"}'

# Get a PR
curl -L \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${INSTALLATION_TOKEN}"\
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/EmilyBourne/pyccel/pulls/24

# ["mergeable_state"] == "clean"
# ["head"]["sha"]

# Create a workflow dispatch event (https://docs.github.com/en/rest/actions/workflows?apiVersion=2022-11-28)
#curl -L \
#  -X POST \
#  -H "Accept: application/vnd.github+json" \
#  -H "Authorization: Bearer ${INSTALLATION_TOKEN}"\
#  -H "X-GitHub-Api-Version: 2022-11-28" \
#  https://api.github.com/repos/EmilyBourne/pyccel/actions/workflows/lint.yml/dispatches \
#  -d '{"ref":"devel","inputs":{"python_version":"3.8", "ref":"e02d795840dea22ee2bae0c1c67d833ab1755fb8"}}'

# Inspect comments (https://docs.github.com/en/rest/issues/comments?apiVersion=2022-11-28)
#curl -L \
#  -H "Accept: application/vnd.github+json" \
#  -H "Authorization: Bearer ${INSTALLATION_TOKEN}"\
#  -H "X-GitHub-Api-Version: 2022-11-28" \
#  https://api.github.com/repos/EmilyBourne/pyccel/issues/24/comments
#
#curl -L \
#  -X DELETE \
#  -H "Accept: application/vnd.github+json" \
#  -H "Authorization: Bearer ${INSTALLATION_TOKEN}"\
#  -H "X-GitHub-Api-Version: 2022-11-28" \
#  https://api.github.com/repos/EmilyBourne/pyccel/issues/comments/1575476719

 #Request reviews : https://docs.github.com/en/rest/pulls/review-requests?apiVersion=2022-11-28

# curl -L \
#  -H "Accept: application/vnd.github+json" \
#  -H "Authorization: Bearer ${INSTALLATION_TOKEN}"\
#  -H "X-GitHub-Api-Version: 2022-11-28" \
#  https://api.github.com/orgs/pyccel/members/EmilyBourne

# curl -L \
#  -H "Accept: application/vnd.github+json" \
#  -H "Authorization: Bearer ${INSTALLATION_TOKEN}"\
#  -H "X-GitHub-Api-Version: 2022-11-28" \
#  https://api.github.com/orgs/pyccel/teams/pyccel-dev



#curl -L \
#  -X PATCH \
#  -H "Accept: application/vnd.github+json" \
#  -H "Authorization: Bearer ${INSTALLATION_TOKEN}"\
#  -H "X-GitHub-Api-Version: 2022-11-28" \
#  https://api.github.com/repos/EmilyBourne/pyccel/pulls/24 \
#  -d '{"draft": false}'
