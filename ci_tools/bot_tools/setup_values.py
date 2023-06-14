from .github_api_interactions import GitHubAPIInteractions


def get_pr_id(possible_pr_ids):
    GAI = GitHubAPIInteractions()
    for p in possible_pr_ids:
        n = p['number']
        pr_details = GAI.get_pr_details(n)
        if pr_details["state"] == "open":
            return n
    return possible_pr_ids[0]['number']
