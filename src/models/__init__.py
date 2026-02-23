"""Model definitions for attribute extraction."""

from src.models.attribute_extractor import BaseAttributeExtractor
from src.models.gpt4o_extractor import GPT4oExtractor
from src.models.llava_extractor import LLaVAExtractor
from src.models.bert_extractor import BERTExtractor
from src.models.rule_based import RuleBasedExtractor

__all__ = [
    "BaseAttributeExtractor",
    "GPT4oExtractor",
    "LLaVAExtractor",
    "BERTExtractor",
    "RuleBasedExtractor",
]
