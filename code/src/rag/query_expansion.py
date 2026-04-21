"""
Query Expansion for Trilingual Legal Search
Expands queries with Arabic, French, and English legal synonyms
Improves recall for multilingual Lebanese legal corpus
"""

import json
from pathlib import Path
from typing import List, Set
from loguru import logger


class TrilingualQueryExpander:
    """
    Expand legal queries with synonyms in Arabic, French, and English.

    Addresses Research Sub-Question 2:
    "How can multi-agent systems effectively handle trilingual Lebanese legal texts?"
    """

    def __init__(self, synonyms_file: str = "./data/legal_synonyms.json"):
        self.synonyms_file = Path(synonyms_file)
        self.synonyms = self._load_synonyms()

        logger.info(f"Query expander loaded with {len(self.synonyms)} synonym sets")

    def _load_synonyms(self) -> dict:
        """Load legal synonyms from JSON file."""

        if self.synonyms_file.exists():
            try:
                with open(self.synonyms_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load synonyms: {e}")
                return self._get_default_synonyms()
        else:
            logger.info("Synonyms file not found, using defaults")
            # Create default file
            default_synonyms = self._get_default_synonyms()
            self._save_synonyms(default_synonyms)
            return default_synonyms

    def _get_default_synonyms(self) -> dict:
        """Get default Lebanese legal synonyms."""

        return {
            # Civil Liability
            "مسؤولية": ["مسؤولية", "responsibility", "responsabilité", "التزام", "liability", "obligation"],
            "التزام": ["التزام", "obligation", "مسؤولية", "commitment", "engagement"],

            # Contract Law
            "عقد": ["عقد", "contract", "contrat", "اتفاق", "agreement", "convention"],
            "اتفاق": ["اتفاق", "agreement", "accord", "عقد", "convention"],
            "اتفاقية": ["اتفاقية", "agreement", "convention", "عقد"],

            # Breach & Violation
            "خرق": ["خرق", "breach", "violation", "انتهاك", "rupture", "infraction"],
            "انتهاك": ["انتهاك", "violation", "infraction", "خرق", "breach"],
            "مخالفة": ["مخالفة", "violation", "contravention", "خرق"],

            # Compensation & Damages
            "تعويض": ["تعويض", "compensation", "indemnisation", "damages", "dédommagement"],
            "ضرر": ["ضرر", "damage", "dommage", "harm", "prejudice", "أذى"],
            "أضرار": ["أضرار", "damages", "dommages", "losses", "ضرر"],

            # Parties
            "طرف": ["طرف", "party", "partie", "جهة", "side"],
            "متعاقد": ["متعاقد", "contractor", "contractant", "contracting party", "طرف"],

            # Rights & Obligations
            "حق": ["حق", "right", "droit", "entitlement"],
            "واجب": ["واجب", "duty", "obligation", "devoir", "التزام"],
            "حقوق": ["حقوق", "rights", "droits"],

            # Legal Actions
            "دعوى": ["دعوى", "lawsuit", "action", "claim", "procès"],
            "دعاوى": ["دعاوى", "lawsuits", "actions", "دعوى"],
            "قضية": ["قضية", "case", "affaire", "matter", "cause"],

            # Sale & Purchase
            "بيع": ["بيع", "sale", "vente", "selling"],
            "شراء": ["شراء", "purchase", "achat", "buying"],
            "بضائع": ["بضائع", "goods", "marchandises", "merchandise", "سلع"],

            # Employment
            "عمل": ["عمل", "work", "employment", "travail", "وظيفة"],
            "موظف": ["موظف", "employee", "employé", "worker", "عامل"],
            "صاحب_عمل": ["صاحب عمل", "employer", "employeur", "مشغل"],

            # Fraud & Deception
            "غش": ["غش", "fraud", "fraude", "deception", "تدليس"],
            "تدليس": ["تدليس", "fraud", "deception", "fraude", "غش"],
            "احتيال": ["احتيال", "fraud", "escroquerie", "deception"],

            # Payment
            "دفع": ["دفع", "payment", "paiement", "سداد"],
            "سداد": ["سداد", "payment", "paiement", "دفع"],
            "ثمن": ["ثمن", "price", "prix", "سعر", "cost"],

            # Delivery
            "تسليم": ["تسليم", "delivery", "livraison", "remise"],
            "استلام": ["استلام", "receipt", "réception", "receiving"],

            # Termination
            "فسخ": ["فسخ", "termination", "résiliation", "cancellation", "إلغاء"],
            "إنهاء": ["إنهاء", "termination", "cessation", "ending", "فسخ"],
            "إلغاء": ["إلغاء", "cancellation", "annulation", "فسخ"],

            # Legal Terms
            "قانون": ["قانون", "law", "loi", "legislation", "تشريع"],
            "مادة": ["مادة", "article", "clause", "provision", "بند"],
            "محكمة": ["محكمة", "court", "tribunal", "cour"],
        }

    def _save_synonyms(self, synonyms: dict):
        """Save synonyms to JSON file."""

        self.synonyms_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.synonyms_file, "w", encoding="utf-8") as f:
            json.dump(synonyms, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(synonyms)} synonym sets to {self.synonyms_file}")

    def expand_query(self, query: str, max_expansions: int = 3) -> str:
        """
        Expand query with trilingual synonyms.

        Args:
            query: Original query
            max_expansions: Maximum synonyms to add per term

        Returns:
            Expanded query with synonyms
        """

        expanded_terms: Set[str] = set()
        query_lower = query.lower()

        # Find matching terms
        for term, synonyms in self.synonyms.items():
            # Check if term appears in query
            if term in query_lower or any(syn in query_lower for syn in synonyms):
                # Add limited synonyms
                expanded_terms.update(synonyms[:max_expansions])

        # Build expanded query
        if expanded_terms:
            expansion = " ".join(list(expanded_terms)[:max_expansions * 2])
            expanded_query = f"{query} {expansion}"

            logger.debug(
                f"Query expanded: '{query[:50]}...' → +{len(expanded_terms)} synonyms"
            )

            return expanded_query

        return query

    def expand_queries(self, queries: List[str], max_expansions: int = 3) -> List[str]:
        """
        Expand multiple queries.

        Args:
            queries: List of queries
            max_expansions: Max synonyms per term

        Returns:
            List of expanded queries
        """

        return [self.expand_query(q, max_expansions) for q in queries]

    def get_synonyms_for_term(self, term: str) -> List[str]:
        """Get all synonyms for a specific term."""

        term_lower = term.lower()

        for key, synonyms in self.synonyms.items():
            if key == term_lower or term_lower in synonyms:
                return synonyms

        return []

    def add_custom_synonyms(self, term: str, synonyms: List[str]):
        """Add custom synonym set."""

        self.synonyms[term.lower()] = synonyms
        self._save_synonyms(self.synonyms)

        logger.info(f"Added synonyms for '{term}': {synonyms}")


# Global expander instance
_expander = None


def get_query_expander() -> TrilingualQueryExpander:
    """Get global query expander instance."""
    global _expander
    if _expander is None:
        _expander = TrilingualQueryExpander()
    return _expander


def expand_query(query: str, max_expansions: int = 3) -> str:
    """Convenience function for query expansion."""
    expander = get_query_expander()
    return expander.expand_query(query, max_expansions)


if __name__ == "__main__":
    # Test query expansion
    print("\n" + "=" * 60)
    print("TESTING TRILINGUAL QUERY EXPANSION")
    print("=" * 60)

    expander = TrilingualQueryExpander()

    test_queries = [
        "ما هي المسؤولية المدنية للموظف؟",
        "خرق عقد البيع",
        "التعويض عن الأضرار",
        "حقوق المشتري عند عدم التسليم",
    ]

    for query in test_queries:
        expanded = expander.expand_query(query, max_expansions=3)

        print(f"\nOriginal:  {query}")
        print(f"Expanded:  {expanded}")
        print(f"Gain:      +{len(expanded) - len(query)} characters")

    print("\n" + "=" * 60)

    # Show some synonym sets
    print("\nExample Synonym Sets:")
    for term in ["مسؤولية", "عقد", "خرق", "تعويض"]:
        synonyms = expander.get_synonyms_for_term(term)
        print(f"\n{term}:")
        print(f"  → {', '.join(synonyms[:5])}...")

    print("\n" + "=" * 60 + "\n")
