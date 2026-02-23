"""Tests for the product feature parser."""

import pytest

from src.data.feature_parser import (
    parse_feature_string,
    normalize_attributes,
)


class TestParseFeatureString:
    """Test raw feature string parsing."""

    def test_basic_parsing(self):
        raw = "Color:Beige | Material:Polyester | Style:Contemporary"
        result = parse_feature_string(raw)
        assert result == {
            "Color": "Beige",
            "Material": "Polyester",
            "Style": "Contemporary",
        }

    def test_empty_string(self):
        assert parse_feature_string("") == {}
        assert parse_feature_string(None) == {}

    def test_single_feature(self):
        raw = "Color:Red"
        result = parse_feature_string(raw)
        assert result == {"Color": "Red"}

    def test_extra_whitespace(self):
        raw = "  Color : Beige  |  Material : Wood  "
        result = parse_feature_string(raw)
        assert result["Color"] == "Beige"
        assert result["Material"] == "Wood"

    def test_colon_in_value(self):
        raw = "Dimensions:24 x 36 x 12:inches"
        result = parse_feature_string(raw)
        assert "Dimensions" in result

    def test_missing_value(self):
        raw = "Color: | Material:Wood"
        result = parse_feature_string(raw)
        assert "Material" in result


class TestNormalizeAttributes:
    """Test attribute normalization."""

    def test_color_normalization(self):
        raw = {"Color": "Beige", "Style": "Contemporary"}
        result = normalize_attributes(raw)
        assert result["color_family"] == "beige"
        assert result["style"] == "contemporary"

    def test_boolean_normalization(self):
        raw = {"Assembly Required": "Yes"}
        result = normalize_attributes(raw)
        assert result["assembly_required"] is True

    def test_boolean_no(self):
        raw = {"Assembly Required": "No"}
        result = normalize_attributes(raw)
        assert result["assembly_required"] is False

    def test_unknown_keys_ignored(self):
        raw = {"Unknown Key": "Some Value", "Color": "Red"}
        result = normalize_attributes(raw)
        assert result["color_family"] == "red"

    def test_all_none_when_empty(self):
        result = normalize_attributes({})
        for value in result.values():
            assert value is None

    def test_material_mapping(self):
        raw = {"Material": "Solid Oak Wood"}
        result = normalize_attributes(raw)
        assert result["primary_material"] == "solid oak wood"
