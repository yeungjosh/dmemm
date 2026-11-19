"""Unit tests for SentimentAnalyzer and the read-only API endpoints.

The demo analyzer ships with untrained weights, so these tests check structure
and invariants (shapes, probabilities, aggregation maths), not predicted labels.

Run from the repo root:  python -m pytest tests -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app', 'backend'))
from sentiment_analyzer import SentimentAnalyzer  # noqa: E402
from app import app  # noqa: E402


@pytest.fixture(scope='module')
def analyzer():
    return SentimentAnalyzer()


def _result(pos, neg, neu):
    return {'probabilities': {'Positive': pos, 'Negative': neg, 'Neutral': neu}}


def test_one_result_per_whitespace_separated_word(analyzer):
    results = analyzer.analyze_text('I love   this\tmovie')
    assert [r['word'] for r in results] == ['i', 'love', 'this', 'movie']


def test_empty_text_gives_no_results(analyzer):
    assert analyzer.analyze_text('') == []
    assert analyzer.analyze_text('   ') == []


def test_unknown_words_do_not_crash(analyzer):
    results = analyzer.analyze_text('zzyzx qwertyuiop')
    assert len(results) == 2


def test_result_fields_and_probabilities(analyzer):
    for r in analyzer.analyze_text('the food was great'):
        assert r['sentiment'] in {'Positive', 'Negative', 'Neutral'}
        assert r['tag'] in {'T-POS', 'T-NEG', 'T-NEU', 'O'}
        assert 0.0 <= r['confidence'] <= 1.0
        assert set(r['probabilities']) == {'Positive', 'Negative', 'Neutral'}
        assert all(0.0 <= p <= 1.0 for p in r['probabilities'].values())


def test_tag_O_is_reported_as_neutral(analyzer):
    assert analyzer.tag_names['O'] == 'Neutral'
    assert analyzer.tag_names['T-NEU'] == 'Neutral'


def test_analysis_is_deterministic(analyzer):
    assert analyzer.analyze_text('I love this') == analyzer.analyze_text('I love this')


def test_overall_of_empty_results_is_neutral_zero(analyzer):
    assert analyzer.get_overall_sentiment([]) == {'sentiment': 'Neutral', 'confidence': 0.0}


def test_overall_of_all_zero_probabilities_is_neutral_zero(analyzer):
    out = analyzer.get_overall_sentiment([_result(0, 0, 0)])
    assert out == {'sentiment': 'Neutral', 'confidence': 0.0}


def test_overall_picks_dominant_and_normalises(analyzer):
    out = analyzer.get_overall_sentiment([_result(0.6, 0.2, 0.2), _result(0.8, 0.1, 0.1)])
    assert out['sentiment'] == 'Positive'
    assert out['confidence'] == pytest.approx(1.4 / 2.0)
    assert sum(out['scores'].values()) == pytest.approx(1.0)


def test_overall_negative_dominates(analyzer):
    out = analyzer.get_overall_sentiment([_result(0.1, 0.8, 0.1)])
    assert out['sentiment'] == 'Negative'


@pytest.fixture
def client():
    return app.test_client()


def test_examples_endpoint(client):
    body = client.get('/api/examples').get_json()
    assert body['success'] is True
    assert body['examples']
    assert all({'text', 'category'} <= set(e) for e in body['examples'])


def test_index_serves_frontend(client):
    r = client.get('/')
    assert r.status_code == 200
    assert b'<html' in r.data.lower()
    r.close()
