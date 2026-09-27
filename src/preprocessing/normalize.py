"""Deterministic text normalization helpers for entity resolution."""

import math
import unicodedata


def _is_missing(value):
	if value is None:
		return True
	if isinstance(value, float) and math.isnan(value):
		return True
	try:
		return bool(value != value)
	except (TypeError, ValueError):
		return False


def _normalize_text(value):
	if _is_missing(value):
		return ""

	normalized = unicodedata.normalize("NFC", str(value)).lower()
	characters = []
	for character in normalized:
		category = unicodedata.category(character)
		if character.isspace():
			characters.append(" ")
		elif category[0] in {"L", "M", "N"}:
			characters.append(character)
		else:
			characters.append(" ")
	return " ".join("".join(characters).split())


def normalize_name(value):
	"""Normalize a business name while preserving Unicode text."""
	return _normalize_text(value)


def normalize_address(value):
	"""Normalize a business address while preserving Unicode text."""
	return _normalize_text(value)


def normalize_country(value):
	"""Normalize a country value using the same deterministic text rules."""
	return _normalize_text(value)


if __name__ == "__main__":
	examples = {
		"English name": "Acme Coffee Co.",
		"Punctuation": "O'Reilly & Sons, LLC.",
		"Accented name": "Café Société",
		"Indian-language name": "श्री गणेश ट्रेडर्स",
		"Address": "12-B, MG Road, Bengaluru, Karnataka 560001",
		"Empty value": "",
	}

	for label, example in examples.items():
		normalizer = normalize_address if label == "Address" else normalize_name
		print(f"{label}: {normalizer(example)!r}")

	assert normalize_name(None) == ""
	assert normalize_name(float("nan")) == ""
	assert normalize_name(" Café  Société! ") == "café société"
	assert normalize_name("श्री गणेश ट्रेडर्स") == "श्री गणेश ट्रेडर्स"
	assert normalize_country("  United   States ") == "united states"
