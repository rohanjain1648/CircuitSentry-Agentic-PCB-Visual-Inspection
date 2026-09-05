import requests


def poll_run_status(api_base_url: str, run_id: str, requests_get=requests.get) -> dict:
    """Return the latest recorded state of a run.

    ``regions`` is carried through (from the same GET /runs/{id} detail
    response) so the caller can find the bbox that triggered a recapture
    without a second round trip.
    """
    response = requests_get(f"{api_base_url}/runs/{run_id}", timeout=10)
    response.raise_for_status()
    items = response.json()["items"]
    latest = max(items, key=lambda i: int(i["seq"]))
    return {
        "seq": int(latest["seq"]),
        "next_action": latest["next_action"],
        "regions": latest.get("regions", []),
    }
