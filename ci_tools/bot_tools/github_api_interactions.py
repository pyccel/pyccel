import requests


class GitHubAPIInteractions:
    def __init__(self, repo, install_token):
        self._org, self._repo = repo.split('/')
        self._headers={"Accept": "application/vnd.github+json",
                 "Authorization:", f"Bearer {install_token}",
                 "X-GitHub-Api-Version": "2022-11-28"}

    def _post_request(self, method, url, json=None):
        return requests.request(method, url, json=json, headers=self._headers)

    def check_runs(self, commit):
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/commits/{commit}/check-runs"
        return self._post_request("GET", url)

    def create_run(self, commit, name, workflow_url):
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/check-runs"
        json = {"name": name,
                "head_sha": commit,
                "status": "in_progress",
                "details_url": workflow_url}
        return self._post_request("POST", url, json)

    def update_run(self, run_id, **kwargs):
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/check-runs/{run_id}"
        return self._post_request("POST", url, kwargs)

    def get_pr_details(self, pr_id):
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/pulls/{pr_id}"
        return self._post_request("GET", url)

    def run_workflow(self, filename, inputs):
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/actions/workflows/{filename}/dispatches"
        json = {"ref": "devel",
                "inputs": str(inputs)}
        return self._post_request("POST", url, json)

    def get_comments(self, pr_id):
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/issues/{pr_id}/comments"
        return self._post_request("GET", url)
