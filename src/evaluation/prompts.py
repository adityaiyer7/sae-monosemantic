LLM_JUDGE_CLASSIFICATION_SYSTEM = """\
You are an expert in mechanistic interpretability of neural networks, \
specifically Sparse Autoencoders (SAEs). Your task is to label what concept \
or pattern a single SAE feature detects based on example activations.

You will receive a numbered list of context strings. In each string the \
activating token is wrapped in **double asterisks**. All examples come from \
the same SAE feature.

Instructions:
1. Read every example carefully.
2. Think step-by-step: identify what the highlighted tokens share. Consider \
lexical identity, morphology, semantic role, syntactic position, surrounding \
context, and topic domain — in that order of priority.
3. Be specific. A good label names the exact lexical item, grammatical \
construction, or semantic category. A bad label is vague (e.g. "common \
English words", "short word fragments", "frequently used tokens"). If you \
find yourself writing a generic description, look harder for a real pattern \
or conclude POLYSEMANTIC.
4. Produce a short, specific label (2-8 words) that captures the pattern.
5. If the highlighted tokens share no coherent lexical, syntactic, or \
semantic pattern — only surface-level properties like being common or \
short — output exactly: POLYSEMANTIC

You MUST end your response with a line in this exact format:
Label: <your label here>"""

LLM_JUDGE_CLASSIFICATION_USER = """\
Here are four worked examples, then the real task.

--- Example 1 ---
Context strings:
1. the river was **flowing** quickly downstream
2. tears were **streaming** down her face
3. lava **pouring** from the volcano

Reasoning: All highlighted tokens are verbs describing the continuous movement \
of a liquid or liquid-like substance. They share a semantic role (motion) and \
a domain (fluids). The pattern is liquid-motion verbs.
Label: Liquid motion verbs

--- Example 2 ---
Context strings:
1. She went to **Paris** last summer
2. flights from **London** to New York
3. the capital **Berlin** hosted the event

Reasoning: Every highlighted token is a European capital city used as a proper \
noun in a travel or geographic context. They share lexical identity (city names) \
and geographic scope (Europe).
Label: European capital cities

--- Example 3 ---
Context strings:
1. the **,** separated list of items
2. red **,** blue **,** and green
3. however **,** the results were

Reasoning: The highlighted token is always a comma. It appears in enumerations \
and after discourse markers. The feature detects commas.
Label: Comma punctuation

--- Example 4 (POLYSEMANTIC) ---
Context strings:
1. the **Gam** ing industry has grown
2. he **Talk** ed about the weather
3. **Econom** ic policies affect everyone
4. **Touch** screen devices are everywhere
5. **Hum** an rights violations reported

Reasoning: The highlighted tokens are subword fragments of unrelated nouns, \
verbs, and adjectives spanning different topics (gaming, speech, economics, \
technology, human rights). They share no lexical, syntactic, or semantic \
pattern beyond being common English word prefixes. This is polysemantic.
Label: POLYSEMANTIC

--- Real Task ---
Context strings:
{context_strings}

Reasoning:"""
