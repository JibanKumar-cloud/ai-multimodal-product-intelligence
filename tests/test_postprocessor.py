"""Tests for the inference postprocessor."""

import pytest

from src.inference.postprocessor import PostProcessor


class TestPostProcessor:
    """Test output parsing and normalization."""

    def setup_method(self):
        self.pp = PostProcessor()

    def test_parse_valid_json(self):
        raw = '{"style": "modern", "color_family": "blue"}'
        result = self.pp.process(raw)
        assert result["style"] == "modern"
        assert result["color_family"] == "blue"

    def test_parse_json_with_markdown(self):
        raw = '```json\n{"style": "modern"}\n```'
        result = self.pp.process(raw)
        assert result["style"] == "modern"

    def test_parse_json_with_preamble(self):
        raw = 'Here are the attributes:\n{"style": "modern", "color_family": "gray"}'
        result = self.pp.process(raw)
        assert result["style"] == "modern"

    def test_normalize_color_aliases(self):
        result = self.pp.process({"color_family": "grey"})
        assert result["color_family"] == "gray"

    def test_normalize_color_cream(self):
        result = self.pp.process({"color_family": "cream"})
        assert result["color_family"] == "white"

    def test_normalize_style_aliases(self):
        result = self.pp.process({"style": "mid century"})
        assert result["style"] == "mid-century modern"

    def test_normalize_room_type_string_to_list(self):
        result = self.pp.process({"room_type": "living room, bedroom"})
        assert isinstance(result["room_type"], list)
        assert "living room" in result["room_type"]
        assert "bedroom" in result["room_type"]

    def test_normalize_assembly_boolean(self):
        result = self.pp.process({"assembly_required": "Yes"})
        assert result["assembly_required"] is True

        result = self.pp.process({"assembly_required": "No"})
        assert result["assembly_required"] is False

    def test_empty_dict_returns_schema(self):
        result = self.pp.process({})
        assert "style" in result
        assert "primary_material" in result
        assert "color_family" in result

    def test_invalid_string_uses_fallback(self):
        raw = "style is modern and color is blue"
        result = self.pp.process(raw)
        assert result["style"] == "modern"
        assert result["color_family"] == "blue"

    def test_completely_unparseable(self):
        raw = "gibberish that means nothing"
        result = self.pp.process(raw)
        assert isinstance(result, dict)
        # Should return empty result, not crash
