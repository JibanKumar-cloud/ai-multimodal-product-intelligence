"""Tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import (
    compute_attribute_f1,
    compute_exact_match,
    compute_weighted_f1,
    compute_full_evaluation,
)


class TestAttributeF1:
    """Test per-attribute F1 computation."""

    def test_perfect_predictions(self):
        preds = [{"style": "modern", "color_family": "blue"}]
        gt = [{"style": "modern", "color_family": "blue"}]
        result = compute_attribute_f1(preds, gt, ["style", "color_family"])
        assert result["style"]["f1"] == 1.0
        assert result["color_family"]["f1"] == 1.0

    def test_all_wrong(self):
        preds = [{"style": "rustic", "color_family": "red"}]
        gt = [{"style": "modern", "color_family": "blue"}]
        result = compute_attribute_f1(preds, gt, ["style", "color_family"])
        assert result["style"]["f1"] == 0.0
        assert result["color_family"]["f1"] == 0.0

    def test_partial_match(self):
        preds = [
            {"style": "modern"},
            {"style": "modern"},
            {"style": "rustic"},
        ]
        gt = [
            {"style": "modern"},
            {"style": "rustic"},
            {"style": "rustic"},
        ]
        result = compute_attribute_f1(preds, gt, ["style"])
        # 1 TP (first), 1 FP+FN (second), 1 TP (third)
        assert 0 < result["style"]["f1"] < 1.0

    def test_none_ground_truth_skipped(self):
        preds = [{"style": "modern"}, {"style": "modern"}]
        gt = [{"style": "modern"}, {"style": None}]
        result = compute_attribute_f1(preds, gt, ["style"])
        assert result["style"]["f1"] == 1.0
        assert result["style"]["support"] == 1

    def test_multi_label_attribute(self):
        preds = [{"room_type": ["living room", "bedroom"]}]
        gt = [{"room_type": ["living room", "office"]}]
        result = compute_attribute_f1(preds, gt, ["room_type"])
        # 1 TP (living room), 1 FP (bedroom), 1 FN (office)
        assert result["room_type"]["precision"] == 0.5
        assert result["room_type"]["recall"] == 0.5

    def test_case_insensitive(self):
        preds = [{"style": "Modern"}]
        gt = [{"style": "modern"}]
        result = compute_attribute_f1(preds, gt, ["style"])
        assert result["style"]["f1"] == 1.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_attribute_f1([{"a": 1}], [{"a": 1}, {"a": 2}])


class TestExactMatch:
    """Test exact match computation."""

    def test_all_match(self):
        preds = [{"style": "modern", "color_family": "blue"}]
        gt = [{"style": "modern", "color_family": "blue"}]
        assert compute_exact_match(preds, gt) == 1.0

    def test_none_match(self):
        preds = [{"style": "rustic", "color_family": "red"}]
        gt = [{"style": "modern", "color_family": "blue"}]
        assert compute_exact_match(preds, gt) == 0.0

    def test_partial_match_is_not_exact(self):
        preds = [{"style": "modern", "color_family": "red"}]
        gt = [{"style": "modern", "color_family": "blue"}]
        assert compute_exact_match(preds, gt) == 0.0

    def test_empty_returns_zero(self):
        assert compute_exact_match([], []) == 0.0


class TestWeightedF1:
    """Test weighted F1 computation."""

    def test_uniform_weights(self):
        per_attr = {
            "style": {"f1": 0.8, "precision": 0.8, "recall": 0.8, "support": 100},
            "color_family": {"f1": 0.6, "precision": 0.6, "recall": 0.6, "support": 100},
        }
        result = compute_weighted_f1(per_attr, weights={"style": 1.0, "color_family": 1.0})
        assert result == 0.7

    def test_higher_weight_on_style(self):
        per_attr = {
            "style": {"f1": 0.8, "precision": 0.8, "recall": 0.8, "support": 100},
            "color_family": {"f1": 0.4, "precision": 0.4, "recall": 0.4, "support": 100},
        }
        result = compute_weighted_f1(per_attr, weights={"style": 2.0, "color_family": 1.0})
        # (2.0 * 0.8 + 1.0 * 0.4) / 3.0 = 2.0 / 3.0 = 0.6667
        assert abs(result - 0.6667) < 0.001


class TestFullEvaluation:
    """Test end-to-end evaluation."""

    def test_full_pipeline(self):
        preds = [
            {"style": "modern", "primary_material": "wood", "color_family": "brown"},
            {"style": "rustic", "primary_material": "metal", "color_family": "gray"},
        ]
        gt = [
            {"style": "modern", "primary_material": "wood", "color_family": "brown"},
            {"style": "rustic", "primary_material": "wood", "color_family": "gray"},
        ]
        result = compute_full_evaluation(preds, gt, model_name="test", cost_per_1k=1.0)
        assert result["model_name"] == "test"
        assert result["num_examples"] == 2
        assert result["average_f1"] > 0
        assert "per_attribute" in result
