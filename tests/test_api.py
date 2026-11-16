"""Input-validation tests for the Flask API.

Run from the repo root:  python -m pytest tests -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app', 'backend'))
from app import app  # noqa: E402


@pytest.fixture
def client():
    return app.test_client()


@pytest.mark.parametrize('payload', [{}, {'text': ''}, {'text': '   '}])
def test_empty_text_is_400(client, payload):
    assert client.post('/api/analyze', json=payload).status_code == 400


def test_malformed_json_is_400(client):
    r = client.post('/api/analyze', data='{not json', content_type='application/json')
    assert r.status_code == 400


def test_non_json_body_is_400(client):
    assert client.post('/api/analyze', data='hello').status_code == 400


@pytest.mark.parametrize('payload', [[1, 2], {'text': 123}, {'text': ['a']}])
def test_wrong_types_are_400(client, payload):
    assert client.post('/api/analyze', json=payload).status_code == 400


def test_valid_text_succeeds(client):
    r = client.post('/api/analyze', json={'text': 'I love this movie'})
    assert r.status_code == 200
    body = r.get_json()
    assert body['success'] is True and body['words']


def test_health(client):
    assert client.get('/api/health').get_json()['status'] == 'healthy'
