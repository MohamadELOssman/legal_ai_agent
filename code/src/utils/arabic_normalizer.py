"""
Arabic Text Normalization
Normalizes Arabic text for better processing and search
"""

import re
import unicodedata
from typing import List
from loguru import logger


class ArabicTextNormalizer:
    """
    Normalize Arabic text for better processing.

    Handles common Arabic text variations:
    - Different forms of Alef (ا, أ, إ, آ)
    - Different forms of Hamza (ء, ؤ, ئ)
    - Tatweel (ـ) removal
    - Diacritics (تشكيل) removal
    - Extra spaces
    """

    def __init__(self):
        # Arabic character mappings
        self.alef_variations = ['أ', 'إ', 'آ', 'ا']
        self.normalized_alef = 'ا'

        self.hamza_variations = ['ؤ', 'ئ', 'ء']
        self.normalized_hamza = 'ء'

        # Diacritics (Harakat)
        self.diacritics = [
            '\u064B',  # Tanween Fath
            '\u064C',  # Tanween Damm
            '\u064D',  # Tanween Kasr
            '\u064E',  # Fatha
            '\u064F',  # Damma
            '\u0650',  # Kasra
            '\u0651',  # Shadda
            '\u0652',  # Sukun
            '\u0653',  # Maddah
            '\u0654',  # Hamza Above
            '\u0655',  # Hamza Below
            '\u0656',  # Subscript Alef
            '\u0640',  # Tatweel
        ]

        logger.debug("Arabic normalizer initialized")

    def normalize(self, text: str, options: dict = None) -> str:
        """
        Normalize Arabic text.

        Args:
            text: Input text
            options: Normalization options
                - remove_diacritics: bool (default: True)
                - normalize_alef: bool (default: True)
                - normalize_hamza: bool (default: True)
                - remove_tatweel: bool (default: True)
                - remove_extra_spaces: bool (default: True)

        Returns:
            Normalized text
        """

        if not text:
            return text

        if options is None:
            options = {}

        # Default options
        remove_diacritics = options.get("remove_diacritics", True)
        normalize_alef = options.get("normalize_alef", True)
        normalize_hamza = options.get("normalize_hamza", True)
        remove_tatweel = options.get("remove_tatweel", True)
        remove_extra_spaces = options.get("remove_extra_spaces", True)

        normalized = text

        # 1. Remove diacritics (تشكيل)
        if remove_diacritics:
            normalized = self._remove_diacritics(normalized)

        # 2. Normalize Alef variations
        if normalize_alef:
            normalized = self._normalize_alef(normalized)

        # 3. Normalize Hamza variations
        if normalize_hamza:
            normalized = self._normalize_hamza(normalized)

        # 4. Remove Tatweel (ـ)
        if remove_tatweel:
            normalized = self._remove_tatweel(normalized)

        # 5. Remove extra spaces
        if remove_extra_spaces:
            normalized = self._remove_extra_spaces(normalized)

        return normalized

    def _remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics (Harakat)."""

        for diacritic in self.diacritics:
            text = text.replace(diacritic, '')

        return text

    def _normalize_alef(self, text: str) -> str:
        """Normalize different forms of Alef to one form."""

        for alef in self.alef_variations:
            text = text.replace(alef, self.normalized_alef)

        return text

    def _normalize_hamza(self, text: str) -> str:
        """Normalize different forms of Hamza."""

        for hamza in self.hamza_variations:
            text = text.replace(hamza, self.normalized_hamza)

        return text

    def _remove_tatweel(self, text: str) -> str:
        """Remove Tatweel (character elongation)."""
        return text.replace('\u0640', '')

    def _remove_extra_spaces(self, text: str) -> str:
        """Remove extra whitespace."""

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Trim
        text = text.strip()

        return text

    def normalize_for_search(self, text: str) -> str:
        """
        Normalize text for search (aggressive normalization).

        Used for improving search recall.
        """

        return self.normalize(text, {
            "remove_diacritics": True,
            "normalize_alef": True,
            "normalize_hamza": True,
            "remove_tatweel": True,
            "remove_extra_spaces": True,
        })

    def normalize_for_display(self, text: str) -> str:
        """
        Normalize text for display (conservative normalization).

        Preserves more of the original text.
        """

        return self.normalize(text, {
            "remove_diacritics": False,  # Keep diacritics for readability
            "normalize_alef": False,      # Keep original Alef forms
            "normalize_hamza": False,     # Keep original Hamza forms
            "remove_tatweel": True,       # Remove elongation
            "remove_extra_spaces": True,
        })

    def clean_legal_text(self, text: str) -> str:
        """
        Clean legal text for processing.

        Balanced normalization for legal documents.
        """

        return self.normalize(text, {
            "remove_diacritics": True,   # Usually safe for legal text
            "normalize_alef": True,      # Improves matching
            "normalize_hamza": True,     # Improves matching
            "remove_tatweel": True,      # Remove elongation
            "remove_extra_spaces": True,
        })


# Global normalizer instance
_normalizer = None


def get_arabic_normalizer() -> ArabicTextNormalizer:
    """Get global Arabic normalizer instance."""
    global _normalizer
    if _normalizer is None:
        _normalizer = ArabicTextNormalizer()
    return _normalizer


def normalize_arabic(text: str, for_search: bool = True) -> str:
    """
    Convenience function for Arabic normalization.

    Args:
        text: Input text
        for_search: If True, use aggressive normalization for search

    Returns:
        Normalized text
    """

    normalizer = get_arabic_normalizer()

    if for_search:
        return normalizer.normalize_for_search(text)
    else:
        return normalizer.normalize_for_display(text)


if __name__ == "__main__":
    # Test Arabic normalization
    print("\n" + "=" * 60)
    print("TESTING ARABIC TEXT NORMALIZATION")
    print("=" * 60)

    normalizer = ArabicTextNormalizer()

    test_cases = [
        "المَسؤوليّة المَدَنيّة",  # With diacritics
        "الإلتزام القانوني",      # Alef with Hamza
        "مُؤسّسة تجارية",         # Hamza on Waw, Shadda
        "المــــادة",              # With Tatweel
        "العقـــد   والمسؤولية",  # Tatweel + extra spaces
    ]

    print("\nNormalization Examples:\n")

    for i, text in enumerate(test_cases, 1):
        normalized = normalizer.normalize_for_search(text)

        print(f"{i}. Original:    {text}")
        print(f"   Normalized: {normalized}")
        print(f"   Length: {len(text)} → {len(normalized)} ({len(text) - len(normalized):+d})")
        print()

    # Show benefits for search
    print("=" * 60)
    print("SEARCH MATCHING EXAMPLE")
    print("=" * 60)

    query = "المسؤولية"
    variations = [
        "المَسؤوليّة",
        "المسئولية",
        "المسؤوليـــة",
        "المَسْؤُولِيَّة",
    ]

    print(f"\nQuery: {query}")
    print(f"Normalized: {normalizer.normalize_for_search(query)}")
    print("\nVariations that will match after normalization:")

    for var in variations:
        norm_var = normalizer.normalize_for_search(var)
        norm_query = normalizer.normalize_for_search(query)

        match = "✓ MATCH" if norm_var == norm_query else "✗ NO MATCH"

        print(f"  {var}")
        print(f"    → {norm_var}")
        print(f"    {match}\n")

    print("=" * 60 + "\n")
