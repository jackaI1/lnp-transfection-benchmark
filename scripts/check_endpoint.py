"""Bounded smoke check, including the deployed revision's model checksum."""

import argparse
import time

import httpx


def check(url, sha256):
    with httpx.Client(base_url=url, timeout=30) as client:
        for attempt in range(30):
            try:
                ready = client.get("/ready")
                ready.raise_for_status()
                if ready.json()["model_sha256"] != sha256:
                    raise ValueError("Previous revision is still serving")
                break
            except (httpx.HTTPError, ValueError):
                if attempt == 29:
                    raise
                time.sleep(5)
        health = client.get("/health")
        health.raise_for_status()
        response = client.post("/predict", json={"smiles": ["CCN(CC)CCCC", "CCCCN(C)CCO"]})
        response.raise_for_status()
        body = response.json()
        assert body["model_sha256"] == sha256
        assert len(body["predictions"]) == 2
        assert client.post("/predict", json={"smiles": ["invalid"]}).status_code == 422
    print("Healthy revision, 2 predictions, matching model checksum, invalid chemistry rejected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    check(args.url.rstrip("/"), args.sha256)
