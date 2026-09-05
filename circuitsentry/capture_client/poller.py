import requests


def poll_run_status(api_base_url: str, run_id: str, requests_get=requests.get) -> dict:
    response = requests_get(f"{api_base_url}/runs/{run_id}", timeout=10)
    response.raise_for_status()
    items = response.json()["items"]
    latest = max(items, key=lambda i: i["seq"])
    return {"seq": latest["seq"], "next_action": latest["next_action"]}
