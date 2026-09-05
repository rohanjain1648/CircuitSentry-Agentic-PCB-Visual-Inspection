from circuitsentry.capture_client.poller import poll_run_status


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_poll_run_status_returns_latest_item():
    payload = {"items": [
        {"seq": 0, "next_action": "recapture"},
        {"seq": 1, "next_action": "flag_for_approval"},
    ]}

    def fake_get(url, timeout=None):
        assert url == "https://api.example.com/runs/run1"
        return FakeResponse(payload)

    result = poll_run_status("https://api.example.com", "run1", requests_get=fake_get)
    assert result == {"seq": 1, "next_action": "flag_for_approval"}
