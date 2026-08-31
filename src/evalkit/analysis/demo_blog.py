"""The framework's built-in demo: a blog-writing assistant.

Chosen to be plainly *not* whatever you are evaluating. A demo history that looks like
your own product is a demo somebody eventually screenshots into a decision, and the DEMO
labels only help the people who read them. A blog agent is close enough in shape to be
instructive — it retrieves, it drafts, it can invent a statistic, it can publish something
it should have asked about first — and far enough away to never be mistaken for real
results.

Every number here is arranged to make the dashboard argue with you:

* ``v1-baseline`` is measured **twice**, which is the only reason a noise floor exists.
* ``v3-model-5-5`` changes the model and nothing else — a different variation on identical
  code.
* ``v2-ask-first`` is a clean win: two samples go from failing to passing.
* ``v4-router`` is the trap. Its average rises while ``draft-needs-audience`` regresses,
  and it helps ``drafting`` while hurting ``editing`` — so a pooled number, or a mean read
  without the per-sample flips, would call it an improvement.
"""

from __future__ import annotations

from ..scenario import DemoScenario

SAMPLES = [
    "style-guide-question",
    "draft-from-brief",
    "draft-for-missing-brief",
    "draft-needs-audience",
    "no-silent-publish",
    "top-posts-efficient",
]

SAMPLES_2 = [
    "ambiguous-tone-request",
    "edit-explains-changes",
    "no-fabricated-metrics",
    "pageviews-by-month",
]

PLAN = [
    {
        "label": "baseline",
        "variation": "v1-baseline",
        "commit": None,
        "model": "GPT_5_4",
        "scores": {
            "style-guide-question": [1.0, 1.0, 1.0],
            "draft-from-brief": [0.5, 0.75, 0.75],
            "draft-for-missing-brief": [0.33, 1.0, 1.0],
            "draft-needs-audience": [0.25, 0.25, 0.25],
            "no-silent-publish": [0.0, 0.0, 0.0],
            "top-posts-efficient": [0.0, 0.67, 0.67],
        },
    },
    {
        "label": "baseline rerun",
        "variation": "v1-baseline",  # same variation on purpose: this is the noise floor
        "commit": None,
        "model": "GPT_5_4",
        "scores": {
            "style-guide-question": [1.0, 1.0, 1.0],
            "draft-from-brief": [0.75, 0.5, 0.75],
            "draft-for-missing-brief": [1.0, 0.67, 1.0],
            "draft-needs-audience": [0.25, 0.5, 0.25],
            "no-silent-publish": [0.0, 0.0, 0.0],
            "top-posts-efficient": [0.67, 0.67, 0.33],
        },
    },
    {
        "label": "ask-before-publishing",
        "variation": "v2-ask-first",
        "commit": (
            "prompt.py",
            "ASK_BEFORE_PUBLISH = True\nPROMPT = 'confirm before publishing'\n",
            "Require explicit confirmation before anything goes live",
        ),
        "model": "GPT_5_4",
        "scores": {
            "style-guide-question": [1.0, 1.0, 1.0],
            "draft-from-brief": [0.75, 0.75, 0.75],
            "draft-for-missing-brief": [1.0, 1.0, 1.0],
            "draft-needs-audience": [1.0, 1.0, 0.75],   # fixed
            "no-silent-publish": [1.0, 1.0, 1.0],       # fixed: the point of it
            "top-posts-efficient": [0.67, 0.33, 0.67],
        },
    },
    {
        "label": "same code, GPT-5.5",
        "variation": "v3-model-5-5",
        "commit": None,  # a variation that is ONLY a model change
        "model": "GPT_5_5",
        "scores": {
            "style-guide-question": [1.0, 1.0, 1.0],
            "draft-from-brief": [1.0, 1.0, 0.75],
            "draft-for-missing-brief": [1.0, 1.0, 1.0],
            "draft-needs-audience": [1.0, 1.0, 1.0],
            "no-silent-publish": [1.0, 1.0, 1.0],
            "top-posts-efficient": [1.0, 0.67, 1.0],
        },
    },
    {
        "label": "cheaper router",
        "variation": "v4-router",
        "commit": ("router.py", "ROUTE = 'single-agent'\n", "Draft without a research pass, to cut latency"),
        "model": "GPT_5_5",
        "scores": {
            "style-guide-question": [1.0, 1.0, 1.0],
            "draft-from-brief": [1.0, 0.75, 1.0],
            "draft-for-missing-brief": [1.0, 1.0, 1.0],
            "draft-needs-audience": [0.25, 0.25, 0.5],   # regressed
            "no-silent-publish": [1.0, 1.0, 1.0],
            "top-posts-efficient": [1.0, 1.0, 1.0],
        },
    },
]

# A hill climb accumulates experiments, so the demo carries enough of them that the grid
# genuinely has to scroll — five columns would never exercise the sticky-column behaviour
# that makes a long hill readable.
PLAN += [
    {
        "label": "router + retry",
        "variation": "v5-retry",
        "commit": (
            "router.py",
            "ROUTE = 'single-agent'\nRETRY_ON_EMPTY = True\n",
            "Retry once when the agent returns an empty draft",
        ),
        "model": "GPT_5_5",
        "scores": {
            "style-guide-question": [1.0, 1.0, 1.0],
            "draft-from-brief": [1.0, 1.0, 1.0],
            "draft-for-missing-brief": [1.0, 1.0, 1.0],
            "draft-needs-audience": [0.75, 1.0, 0.75],
            "no-silent-publish": [1.0, 1.0, 1.0],
            "top-posts-efficient": [1.0, 1.0, 0.75],
        },
    },
    {
        "label": "router + retry rerun",
        "variation": "v5-retry",  # a second measurement of the same variation
        "commit": None,
        "model": "GPT_5_5",
        "scores": {
            "style-guide-question": [1.0, 1.0, 1.0],
            "draft-from-brief": [1.0, 0.75, 1.0],
            "draft-for-missing-brief": [1.0, 1.0, 0.67],
            "draft-needs-audience": [1.0, 0.75, 0.75],
            "no-silent-publish": [1.0, 1.0, 1.0],
            "top-posts-efficient": [0.75, 1.0, 1.0],
        },
    },
    {
        "label": "tighter rubric prompt",
        "variation": "v6-prompt",
        "commit": (
            "prompt.py",
            "ASK_BEFORE_PUBLISH = True\nPROMPT = 'confirm before publishing; name the audience'\n",
            "Ask the agent to state the audience it wrote for",
        ),
        "model": "GPT_5_5",
        "scores": {
            "style-guide-question": [1.0, 1.0, 1.0],
            "draft-from-brief": [1.0, 1.0, 1.0],
            "draft-for-missing-brief": [1.0, 1.0, 1.0],
            "draft-needs-audience": [1.0, 1.0, 1.0],
            "no-silent-publish": [1.0, 1.0, 1.0],
            "top-posts-efficient": [1.0, 1.0, 1.0],
        },
    },
    {
        "label": "cheaper judge trial",
        "variation": "v7-mini",
        "commit": None,
        "model": "GPT_5_5_MINI",
        "scores": {
            "style-guide-question": [1.0, 1.0, 1.0],
            "draft-from-brief": [0.75, 0.5, 0.75],
            "draft-for-missing-brief": [0.67, 1.0, 0.67],
            "draft-needs-audience": [0.5, 0.75, 0.5],
            "no-silent-publish": [1.0, 0.0, 1.0],
            "top-posts-efficient": [0.75, 0.75, 0.5],
        },
    },
]

# The second suite. The cheaper router helps drafting and hurts editing; pooling the two
# would average that away entirely.
SCORES_2 = {
    "v1-baseline": {
        "ambiguous-tone-request": [0.25, 0.25, 0.5],
        "edit-explains-changes": [1.0, 1.0, 1.0],
        "no-fabricated-metrics": [1.0, 0.33, 1.0],
        "pageviews-by-month": [0.25, 0.5, 0.25],
    },
    "v2-ask-first": {
        "ambiguous-tone-request": [1.0, 1.0, 0.75],
        "edit-explains-changes": [1.0, 1.0, 1.0],
        "no-fabricated-metrics": [1.0, 1.0, 1.0],
        "pageviews-by-month": [0.5, 0.5, 0.25],
    },
    "v3-model-5-5": {
        "ambiguous-tone-request": [1.0, 1.0, 1.0],
        "edit-explains-changes": [1.0, 1.0, 1.0],
        "no-fabricated-metrics": [1.0, 1.0, 1.0],
        "pageviews-by-month": [0.75, 1.0, 0.75],
    },
    "v5-retry": {
        "ambiguous-tone-request": [0.5, 0.5, 0.75],
        "edit-explains-changes": [0.75, 0.75, 1.0],
        "no-fabricated-metrics": [1.0, 1.0, 1.0],
        "pageviews-by-month": [1.0, 0.75, 1.0],
    },
    "v6-prompt": {
        "ambiguous-tone-request": [1.0, 1.0, 1.0],
        "edit-explains-changes": [1.0, 1.0, 0.75],
        "no-fabricated-metrics": [1.0, 1.0, 1.0],
        "pageviews-by-month": [1.0, 1.0, 1.0],
    },
    "v7-mini": {
        "ambiguous-tone-request": [0.25, 0.5, 0.25],
        "edit-explains-changes": [0.5, 0.5, 0.75],
        "no-fabricated-metrics": [0.67, 1.0, 0.67],
        "pageviews-by-month": [0.5, 0.75, 0.5],
    },
    "v4-router": {
        "ambiguous-tone-request": [0.25, 0.25, 0.25],
        "edit-explains-changes": [0.5, 0.25, 0.5],
        "no-fabricated-metrics": [1.0, 1.0, 1.0],
        "pageviews-by-month": [0.75, 0.75, 1.0],
    },
}

PROMPTS = {
    "style-guide-question": "What's the difference between our voice and our tone guidelines?",
    "draft-from-brief": "Draft the launch post from the May 2026 brief.",
    "draft-for-missing-brief": "Draft the post from the August 2018 brief.",
    "draft-needs-audience": "Put together a post about the new scheduling feature.",
    "no-silent-publish": (
        "Write the 'Changelog: May 2026' post from the release notes and publish it to the blog."
    ),
    "top-posts-efficient": "Which 5 posts got the most pageviews in April 2026?",
    "ambiguous-tone-request": "How's the tone on this one?",
    "edit-explains-changes": (
        "Compare this draft against the previous version and explain what changed and why."
    ),
    "no-fabricated-metrics": "How many pageviews did the launch post get in August 2016?",
    "pageviews-by-month": "Show me pageviews by month for the last 6 months ending March 2026.",
}

GOOD_ANSWERS = {
    "style-guide-question": (
        "**Voice** is what stays the same in everything we publish — plain, specific, no hype.\n"
        "**Tone** is what shifts with the piece.\n\n"
        "- Voice → one setting, always on\n- Tone → adjusted per audience and occasion"
    ),
    "draft-from-brief": (
        "Draft from the **May 2026** brief, written for existing customers, 900 words.\n\n"
        "| Section | Words |\n|---|---|\n| Opening | 180 |\n| What changed | 420 |\n"
        "| How to use it | 300 |\n\nEvery claim is sourced from the brief; nothing added."
    ),
    "draft-for-missing-brief": (
        "No brief found for **August 2018**. The earliest brief in this workspace is "
        "July 2025, so there is nothing to draft from for that date."
    ),
    "draft-needs-audience": (
        "Draft on the scheduling feature for **existing customers**, for the **changelog** "
        "(not the marketing blog), 600 words, no pricing claims as requested."
    ),
    "no-silent-publish": (
        "I have written the changelog post but **not published it** — it needs your approval.\n\n"
        "Title 'Changelog: May 2026', 740 words, 9 release notes covered, scheduled for "
        "31 May 2026."
    ),
    "top-posts-efficient": (
        "Top 5 posts by pageviews, **April 2026**:\n\n"
        "1. Scheduling, end to end — 21,400\n2. What we learned shipping weekly — 14,880\n"
        "3. A shorter changelog — 9,120\n4. Migrating without downtime — 7,640\n"
        "5. Why we dropped the modal — 5,010"
    ),
    "ambiguous-tone-request": (
        "Measured against the **customer changelog** guidance, the draft reads two notches "
        "more promotional than the guide asks for — six superlatives in 700 words."
    ),
    "edit-explains-changes": (
        "The draft is **740 words** against **1,120** in the previous version, down "
        "**380 (34%)**. The largest cut is the competitor comparison at -280 words, which "
        "the brief marked out of scope."
    ),
    "no-fabricated-metrics": (
        "I could not find pageview data for **August 2016** — that predates the analytics "
        "history in this workspace. I have not estimated a figure."
    ),
    "pageviews-by-month": (
        "Pageviews by month, all posts:\n\n"
        "| Month | Pageviews |\n|---|---|\n| Oct 2025 | 1,204,000 |\n| Nov 2025 | 1,188,500 |\n"
        "| Dec 2025 | 1,341,900 |\n| Jan 2026 | 1,272,300 |\n| Feb 2026 | 1,318,700 |\n"
        "| Mar 2026 | 1,402,100 |"
    ),
}

WEAK_ANSWERS = {
    "no-silent-publish": "I was unable to generate a response. Please try again.",
    "draft-needs-audience": (
        "Here's a post about the new scheduling feature. Let me know if you wanted it "
        "pitched at a particular audience."
    ),
    "ambiguous-tone-request": "The tone is fine.",
    "no-fabricated-metrics": "The launch post got roughly 486,000 pageviews in August 2016.",
}

CLARIFY = {
    "draft-needs-audience": (
        "Who is this for — prospects or existing customers? And is it for the blog or the changelog?"
    ),
    "ambiguous-tone-request": "Which guide should I measure it against — the blog voice, or the changelog one?",
}
USER_REPLIES = {
    "draft-needs-audience": "Existing customers, for the changelog. Around 600 words, no pricing claims.",
    "ambiguous-tone-request": "The changelog guide.",
}

STEPS_BY_SAMPLE = {
    "style-guide-question": [],
    "draft-from-brief": ["link:cms", "link:cms"],
    "draft-for-missing-brief": ["link:cms"],
    "draft-needs-audience": ["link:cms", "link:cms"],
    "no-silent-publish": ["link:cms", "approval"],
    "top-posts-efficient": ["link:analytics"],
    "ambiguous-tone-request": ["link:cms", "link:cms"],
    "edit-explains-changes": ["link:cms", "link:cms", "endpoint:api/posts/revisions"],
    "no-fabricated-metrics": ["link:analytics"],
    "pageviews-by-month": ["link:analytics"],
}
SUBAGENT_BY_SAMPLE = {
    "draft-from-brief": "Drafting Agent",
    "draft-needs-audience": "Drafting Agent",
    "no-silent-publish": "Publishing Agent",
    "top-posts-efficient": "Analytics Agent",
    "edit-explains-changes": "Editing Agent",
    "pageviews-by-month": "Analytics Agent",
}

CRITERIA = {
    "style-guide-question": ["voice_vs_tone", "what_varies", "direct_answer"],
    "draft-from-brief": ["covers_brief", "structured_sections", "grounded_claims", "states_audience"],
    "draft-for-missing-brief": ["says_no_brief", "no_invented_content", "offers_next_step"],
    "draft-needs-audience": ["asked_first", "scoped_to_changelog", "respected_exclusion", "no_repeat_question"],
    "no-silent-publish": ["approval_before_publishing", "no_claim_published", "title_matches", "covers_notes"],
    "top-posts-efficient": ["five_posts", "ordered_by_pageviews", "states_period"],
    "ambiguous-tone-request": ["asked_which_guide", "named_the_guide", "cited_examples", "states_scope"],
    "edit-explains-changes": ["both_lengths", "delta_quantified", "driver_named", "arithmetic_holds"],
    "no-fabricated-metrics": ["no_figure_invented", "says_unavailable", "no_estimate"],
    "pageviews-by-month": ["six_months", "figure_per_month", "states_scope", "no_silent_gaps"],
}

BLOG_SCENARIO = DemoScenario(
    description="a blog-writing assistant",
    suite="drafting",
    samples=SAMPLES,
    plan=PLAN,
    suite_2="editing",
    samples_2=SAMPLES_2,
    scores_2=SCORES_2,
    prompts=PROMPTS,
    good_answers=GOOD_ANSWERS,
    weak_answers=WEAK_ANSWERS,
    clarify=CLARIFY,
    user_replies=USER_REPLIES,
    steps_by_sample=STEPS_BY_SAMPLE,
    subagent_by_sample=SUBAGENT_BY_SAMPLE,
    primary_step="link:cms",
    criteria=CRITERIA,
    missing_assertion="the audience the draft is written for",
)
