from fastapi.testclient import TestClient

from lnp_benchmark.platform import app


def test_team_templates_are_scoped_and_restrict_network():
    with TestClient(app) as client:
        response = client.post("/teams", json={"team": "lipids", "group": "researchers"})
    assert response.status_code == 201
    items = response.json()["items"]
    assert [item["kind"] for item in items] == [
        "Namespace",
        "ResourceQuota",
        "RoleBinding",
        "NetworkPolicy",
        "NetworkPolicy",
    ]
    assert all(item["metadata"].get("namespace", "team-lipids") == "team-lipids" for item in items)
    assert items[2]["subjects"][0]["name"] == "researchers"
    assert items[3]["spec"]["policyTypes"] == ["Ingress", "Egress"]
    assert items[4]["spec"]["egress"][1]["ports"][0]["port"] == 53


def test_team_input_cannot_escape_namespace():
    with TestClient(app) as client:
        for team in ["../admin", "UPPER", "-bad", "a" * 41, ""]:
            assert client.post("/teams", json={"team": team, "group": "x"}).status_code == 422
        assert (
            client.post("/teams", json={"team": "ok", "group": "x", "extra": True}).status_code
            == 422
        )
