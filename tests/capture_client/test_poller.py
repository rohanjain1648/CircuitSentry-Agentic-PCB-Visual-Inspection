from circuitsentry.capture_client.poller import poll_run_status


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_poll_run_status_returns_latest_item():
    regions = [{"region_id": "run1:0", "bbox": [1, 2, 3, 4], "action": "flag_for_approval"}]
    payload = {"items": [
        {"seq": 0, "next_action": "recapture", "regions": []},
        {"seq": 1, "next_action": "flag_for_approval", "regions": regions},
    ]}

    def fake_get(url, timeout=None):
        assert url == "https://api.example.com/runs/run1"
        return FakeResponse(payload)

    result = poll_run_status("https://api.example.com", "run1", requests_get=fake_get)
    assert result == {"seq": 1, "next_action": "flag_for_approval", "regions": regions}


def test_poll_run_status_defaults_regions_when_absent():
    payload = {"items": [{"seq": 0, "next_action": "pass"}]}

    def fake_get(url, timeout=None):
        return FakeResponse(payload)

    result = poll_run_status("https://api.example.com", "run1", requests_get=fake_get)
    assert result["regions"] == []
