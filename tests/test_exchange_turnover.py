"""Tests for the exchange-turnover endpoint, including a brute-force cross-check."""

from __future__ import annotations

import random
from fractions import Fraction

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def parse_fraction(text: str) -> Fraction:
    n, d = text.split("/")
    return Fraction(int(n), int(d))


def fmt(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def brute_force_turnover(payload: dict):
    """Exact turnover inference by explicit exchange-sequence enumeration.

    Every step picks one of the n*m (molecule site, bath site) pairs, so each
    of the (n*m)**steps pick sequences is equally likely; sequences are
    enumerated recursively with the actual heavy-site sets tracked, and the
    running count of "heavy isotope out of the molecule" swaps is carried
    along. Every sequence passing all observation filters adds one to the
    joint count and to the turnover bin of its total outflow count. Returns
    the joint probability of the observations and the conditional
    turnover-count distribution given all of them.
    """
    n, m, h, b, steps = (payload[k] for k in ("n", "m", "h", "b", "steps"))
    filters = {
        obs["step"]: (obs["lower"], obs["upper"])
        for obs in payload.get("observations", [])
    }
    z = 0
    turnover_hits: dict = {}

    def walk(t, mol_heavy, bath_heavy, x, lost):
        nonlocal z
        if t in filters:
            lower, upper = filters[t]
            if not lower <= x <= upper:
                return
        if t == steps:
            z += 1
            turnover_hits[lost] = turnover_hits.get(lost, 0) + 1
            return
        for i in range(n):
            for j in range(m):
                i_heavy = i in mol_heavy
                j_heavy = j < bath_heavy  # bath sites 0..bath_heavy-1 heavy
                if i_heavy == j_heavy:
                    walk(t + 1, mol_heavy, bath_heavy, x, lost)
                elif i_heavy:  # heavy isotope leaves the molecule
                    walk(
                        t + 1,
                        mol_heavy - {i},
                        bath_heavy + 1,
                        x - 1,
                        lost + 1,
                    )
                else:  # heavy isotope enters the molecule
                    walk(
                        t + 1,
                        mol_heavy | {i},
                        bath_heavy - 1,
                        x + 1,
                        lost,
                    )

    walk(0, frozenset(range(h)), b, h, 0)
    probability = Fraction(z, (n * m) ** steps)
    if z == 0:
        return probability, None
    distribution = {
        count: Fraction(count_weight, z)
        for count, count_weight in sorted(turnover_hits.items())
    }
    return probability, distribution


def basic_payload():
    return {
        "n": 2,
        "m": 2,
        "h": 1,
        "b": 1,
        "steps": 2,
        "observations": [{"step": 2, "lower": 2, "upper": 2}],
    }


def test_steps_zero_turnover_is_zero():
    # No exchange happens, so the only possible turnover count is 0; an
    # initial-state observation containing h is certain.
    resp = client.post(
        "/v1/exchange-turnover",
        json={
            "n": 3,
            "m": 2,
            "h": 2,
            "b": 1,
            "steps": 0,
            "observations": [{"step": 0, "lower": 2, "upper": 2}],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "distribution": [{"count": 0, "probability": "1/1"}],
    }


def test_steps_zero_impossible_observation():
    resp = client.post(
        "/v1/exchange-turnover",
        json={
            "n": 3,
            "m": 2,
            "h": 2,
            "b": 1,
            "steps": 0,
            "observations": [{"step": 0, "lower": 0, "upper": 1}],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"probability": "0/1"}


def test_observation_requiring_only_heavy_in_swaps():
    # X_2 = 2 from X_0 = 1 requires two heavy-in swaps and no heavy-out
    # swaps (path 1 -> 1 -> 2, probability 1/2 * 1/4 = 1/8), so turnover is
    # pinned to 0.
    resp = client.post("/v1/exchange-turnover", json=basic_payload())
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/8",
        "distribution": [{"count": 0, "probability": "1/1"}],
    }


def test_round_trip_paths_accumulate_separately():
    # n = m = 2, h = b = 1, two unconditional steps: of the 16 equally
    # likely pair-pick sequences, the 12 ending back at X_2 = 1 split by
    # excursion:
    #   1 -> 0 -> 1 (one heavy out then one heavy in): 2 sequences
    #   1 -> 1 -> 1 (same-type swaps): 4 sequences
    #   1 -> 2 -> 1 (heavy in then heavy out): 8 sequences
    # and X_2 = 0 (down then stay, weight 2) carries turnover 1, while
    # X_2 = 2 (up then stay, weight 2) carries turnover 0. Overall
    # k = 0: 4 + 2 = 6 and k = 1: 2 + 8 + 2 = 10, all over 16. Crucially
    # the round trips end at the same heavy count as the start yet carry
    # different turnover counts than the stay-stay histories.
    resp = client.post(
        "/v1/exchange-turnover",
        json={
            "n": 2,
            "m": 2,
            "h": 1,
            "b": 1,
            "steps": 2,
            "observations": [],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "distribution": [
            {"count": 0, "probability": "3/8"},
            {"count": 1, "probability": "5/8"},
        ],
    }


def test_conditional_round_trip_example():
    # Same chain, conditioning on the round-trip endpoint X_2 = 1 (joint
    # probability 12/16 = 3/4): the endpoint equals the start, yet the
    # excursion histories keep their own turnover counts: k = 0
    # (stay-stay, weight 4) and k = 1 (down-up, weight 2, up-down,
    # weight 8), hence k = 0 with 4/12 = 1/3 and k = 1 with 10/12 = 5/6.
    resp = client.post(
        "/v1/exchange-turnover",
        json={
            "n": 2,
            "m": 2,
            "h": 1,
            "b": 1,
            "steps": 2,
            "observations": [{"step": 2, "lower": 1, "upper": 1}],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "3/4",
        "distribution": [
            {"count": 0, "probability": "1/3"},
            {"count": 1, "probability": "2/3"},
        ],
    }


def test_deterministic_repeated_outflows():
    # n = m = 1, h = 1, b = 0: the only pair swaps the heavy isotope out
    # and back alternately, so every step changes the count and odd steps
    # are heavy-out swaps. Over 4 steps the turnover count is 2 exactly.
    resp = client.post(
        "/v1/exchange-turnover",
        json={"n": 1, "m": 1, "h": 1, "b": 0, "steps": 4, "observations": []},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "probability": "1/1",
        "distribution": [{"count": 2, "probability": "1/1"}],
    }


def test_zero_probability_omits_distribution():
    # From x = 2 with total 2 the chain must move down, so X_1 = 2 is
    # impossible.
    payload = basic_payload()
    payload.update(h=2, b=0, steps=1)
    payload["observations"] = [{"step": 1, "lower": 2, "upper": 2}]
    resp = client.post("/v1/exchange-turnover", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"probability": "0/1"}


def test_step_zero_observation_filters_initial_state():
    payload = basic_payload()
    payload["observations"] = [{"step": 0, "lower": 2, "upper": 2}]
    resp = client.post("/v1/exchange-turnover", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"probability": "0/1"}


def test_randomized_matches_brute_force():
    rng = random.Random(20260928)
    for _ in range(60):
        n = rng.randint(1, 3)
        m = rng.randint(1, 3)
        steps = rng.randint(0, 5)
        obs_count = rng.randint(0, 3)
        observations = []
        for step in rng.sample(range(steps + 1), min(obs_count, steps + 1)):
            lower = rng.randint(0, n)
            upper = rng.randint(lower, n)
            observations.append(
                {"step": step, "lower": lower, "upper": upper}
            )
        payload = {
            "n": n,
            "m": m,
            "h": rng.randint(0, n),
            "b": rng.randint(0, m),
            "steps": steps,
            "observations": observations,
        }
        resp = client.post("/v1/exchange-turnover", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        probability, distribution = brute_force_turnover(payload)
        assert parse_fraction(body["probability"]) == probability
        if probability == 0:
            assert "distribution" not in body
            continue
        got = {
            peak["count"]: parse_fraction(peak["probability"])
            for peak in body["distribution"]
        }
        assert got == distribution
        # Ascending by count, reduced fractions, none zero, normalized.
        counts = [peak["count"] for peak in body["distribution"]]
        assert counts == sorted(counts)
        assert counts and min(counts) >= 0
        for peak in body["distribution"]:
            value = parse_fraction(peak["probability"])
            assert peak["probability"] == fmt(value)
            assert value > 0
        assert sum(got.values(), Fraction(0)) == 1


def test_large_request_stays_exact_and_fast():
    payload = {
        "n": 40,
        "m": 40,
        "h": 20,
        "b": 20,
        "steps": 80,
        "observations": [
            {"step": 20, "lower": 15, "upper": 25},
            {"step": 40, "lower": 10, "upper": 30},
            {"step": 60, "lower": 0, "upper": 40},
            {"step": 80, "lower": 18, "upper": 22},
        ],
    }
    resp = client.post("/v1/exchange-turnover", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    probability = parse_fraction(body["probability"])
    assert 0 < probability < 1
    assert body["probability"] == fmt(probability)
    counts = [peak["count"] for peak in body["distribution"]]
    assert counts == sorted(counts)
    assert counts  # non-empty
    assert min(counts) >= 0 and max(counts) <= 80
    total = sum(
        (parse_fraction(peak["probability"]) for peak in body["distribution"]),
        Fraction(0),
    )
    assert total == 1
    for peak in body["distribution"]:
        value = parse_fraction(peak["probability"])
        assert peak["probability"] == fmt(value)
        assert 0 < value < 1 or value == 1


def test_invalid_payloads_return_422():
    # Reuse the full validation matrix semantics of the trajectory endpoint:
    # representative illegal inputs in every field must yield 422.
    mutations = [
        lambda p: p.update(n=0),
        lambda p: p.update(n=41),
        lambda p: p.update(n=2.0),
        lambda p: p.update(n="2"),
        lambda p: p.pop("n"),
        lambda p: p.update(m=0),
        lambda p: p.update(m=41),
        lambda p: p.update(m="2"),
        lambda p: p.pop("m"),
        lambda p: p.update(h=-1),
        lambda p: p.update(h=3),  # h > n
        lambda p: p.update(h="1"),
        lambda p: p.pop("h"),
        lambda p: p.update(b=-1),
        lambda p: p.update(b=3),  # b > m
        lambda p: p.update(b=False),
        lambda p: p.pop("b"),
        lambda p: p.update(steps=-1),
        lambda p: p.update(steps=81),
        lambda p: p.update(steps=2.5),
        lambda p: p.pop("steps"),
        lambda p: p.pop("observations"),
        lambda p: p.update(
            observations=[{"step": 0, "lower": 0, "upper": 0}] * 9
        ),
        lambda p: p.update(observations={"step": 0}),
        lambda p: p.update(
            observations=[
                {"step": 1, "lower": 0, "upper": 1},
                {"step": 1, "lower": 1, "upper": 2},
            ]
        ),
        lambda p: p["observations"][0].update(step=3),  # step > steps
        lambda p: p["observations"][0].update(upper=3),  # upper > n
        lambda p: p["observations"][0].update(lower=2, upper=1),
        lambda p: p["observations"][0].update(upper="2"),
        lambda p: p["observations"][0].pop("lower"),
        lambda p: p["observations"][0].update(extra=1),
        lambda p: p.update(extra=1),
    ]
    for mutation in mutations:
        payload = basic_payload()
        mutation(payload)
        resp = client.post("/v1/exchange-turnover", json=payload)
        assert resp.status_code == 422, payload


def test_malformed_json_returns_422():
    resp = client.post(
        "/v1/exchange-turnover",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
