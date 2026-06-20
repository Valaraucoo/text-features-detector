"""Feature registry: canonical FeatureSpec definitions for public-dataset tasks."""

from __future__ import annotations

from text_features_detector.models import Feature, FeatureSpec

FEATURE_REGISTRY: dict[Feature, FeatureSpec] = {
    Feature.SENTIMENT_POSITIVE: FeatureSpec(
        name=Feature.SENTIMENT_POSITIVE,
        display_name="Sentiment: Positive",
        criteria=(
            "The text expresses an overall positive evaluation, approval, praise, satisfaction, "
            "or a favourable opinion toward the subject. Indirect praise counts as positive "
            "when the target is presented as valuable, skillful, meaningful, impressive, or admirable."
        ),
        negative_criteria=(
            "The text is negative, neutral, mixed without a clearly positive overall orientation, "
            "or states that positive qualities are missing or lacking. Mentions of positive words "
            "inside a negated or absent context do not count as positive sentiment."
        ),
        evaluation_steps=[
            "Identify the target of evaluation and all sentiment-bearing phrases.",
            "Check whether praise or favourable evaluation is asserted about that target.",
            "Treat negated or missing positive qualities as negative evidence, not positive evidence.",
            "Assign high score only when the overall evaluation is clearly positive.",
        ],
        positive_label_description="positive",
        negative_label_description="negative",
    ),
    Feature.FORMALITY: FeatureSpec(
        name=Feature.FORMALITY,
        display_name="Formality",
        criteria=(
            "The text is written in a formal register: it uses standard grammar, complete or polished "
            "sentence structure, and vocabulary appropriate for professional, academic, news, or official contexts."
        ),
        negative_criteria=(
            "The text is informal or casual: it uses contractions, slang, colloquialisms, chat-like wording, "
            "casual imperatives, fragments, expressive punctuation, or register suited to everyday conversation."
        ),
        evaluation_steps=[
            "Look for contractions (e.g. don't, I'm), slang, or informal vocabulary.",
            "Assess sentence structure: are sentences complete and grammatically standard?",
            "Consider the overall register: would this text be appropriate in professional, "
            "academic, news, or official prose?",
            "Assign high score only when the register is clearly formal; contractions and "
            "casual imperatives are strong informal cues.",
        ],
        positive_label_description="formal",
        negative_label_description="informal",
    ),
    Feature.GRAMMATICAL_ACCEPTABILITY: FeatureSpec(
        name=Feature.GRAMMATICAL_ACCEPTABILITY,
        display_name="Grammatical Acceptability",
        criteria=(
            "The text is grammatically acceptable English: a native speaker would judge it as well-formed, "
            "including syntax, agreement, word order, argument structure, idiomatic preposition/verb use, "
            "and semantic-syntactic compatibility."
        ),
        negative_criteria=(
            "The text is grammatically unacceptable or degraded: it contains syntactic, agreement, word-order, "
            "argument-structure, selectional, or idiomatic usage problems, even if the surface "
            "word order looks plausible."
        ),
        evaluation_steps=[
            "Check subject-verb agreement, tense, articles, prepositions, and morphology.",
            "Check word order, argument structure, verb-object compatibility, and idiomatic usage.",
            "Do not rely only on surface plausibility; consider whether a native speaker would accept the sentence.",
            "Assign high score only when the sentence is clearly acceptable English.",
        ],
        positive_label_description="acceptable",
        negative_label_description="unacceptable",
    ),
}


def get_feature_spec(feature: Feature | str) -> FeatureSpec:
    """Retrieve a FeatureSpec by Feature enum or string value."""
    try:
        return FEATURE_REGISTRY[Feature(feature)]
    except (KeyError, ValueError):
        available = ", ".join(f.value for f in Feature)
        raise KeyError(f"Unknown feature {feature!r}. Available: {available}") from None


def list_features() -> list[Feature]:
    return sorted(FEATURE_REGISTRY)
