LLM_JUDGE_CLASSIFICATION_SYSTEM = """\
You are an expert in mechanistic interpretability of neural networks, \
specifically Sparse Autoencoders (SAEs). Your task is to label what concept \
or pattern a single SAE feature detects based on example activations.

You will receive a numbered list of context strings. In each string the \
activating token is wrapped in **double asterisks**. All examples come from \
the same SAE feature.

Instructions:
1. Read every example carefully.
2. Think step-by-step: identify what the highlighted tokens share (lexical, \
syntactic, semantic, positional, or any other pattern).
3. Produce a short, specific label (2-8 words) that captures the pattern.
4. If the examples are diverse with no coherent pattern, output exactly: \
POLYSEMANTIC - no single pattern detected"""

LLM_JUDGE_CLASSIFICATION_USER = """\
Here are three worked examples, then the real task.

--- Example 1 ---
Context strings:
1. the river was **flowing** quickly downstream
2. tears were **streaming** down her face
3. lava **pouring** from the volcano

Reasoning: All highlighted tokens are verbs describing the continuous movement \
of a liquid or liquid-like substance. The pattern is liquid-motion verbs.
Label: Liquid motion verbs

--- Example 2 ---
Context strings:
1. She went to **Paris** last summer
2. flights from **London** to New York
3. the capital **Berlin** hosted the event

Reasoning: Every highlighted token is a European capital city used as a proper \
noun in a travel or geographic context.
Label: European capital cities

--- Example 3 ---
Context strings:
1. the **,** separated list of items
2. red **,** blue **,** and green
3. however **,** the results were

Reasoning: The highlighted token is always a comma. It appears in enumerations \
and after discourse markers. The feature detects commas.
Label: Comma punctuation

--- Real Task ---
Context strings:
{context_strings}

Reasoning:"""
