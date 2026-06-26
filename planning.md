# TakeMeter — Annotation Planning Document

---

## 1. Community

The community is YouTube viewers who leave public comments on videos. YouTube comments are a good fit for a classification task because the discourse is naturally varied: the same video can attract enthusiastic praise, sharp criticism, neutral questions, and off-topic observations all within the same thread. Comment length, vocabulary, and tone shift dramatically across genres (tutorials, vlogs, music videos, news), giving a classifier meaningful signal to learn from rather than a narrow, homogeneous text type.

YouTube was chosen specifically because comment data is publicly accessible, high-volume, and directly useful — creators and platform tools already depend on understanding audience sentiment, so a working classifier has a real downstream application.

---

## 2. Labels

Three labels are used:

- **Positive** — A comment that expresses approval, enjoyment, gratitude, or enthusiasm toward the video, creator, or topic.
  - "This is the best tutorial I've ever watched — everything clicked immediately!"
  - "Thank you so much for this. I've been struggling with this concept for weeks and you explained it perfectly."

- **Negative** — A comment that expresses disapproval, frustration, criticism, or disappointment toward the video, creator, or topic.
  - "This video was a complete waste of time — nothing was explained clearly."
  - "The background music is way too loud and distracting. Couldn't focus at all."

- **Neutral** — A comment that states a fact, asks a question, or makes an observation without expressing a clear positive or negative emotion.
  - "Came here from the recommended section."
  - "Does this method also work on Windows 11?"

---

## 3. Hard Edge Cases

The most genuinely ambiguous boundary is **positive vs. neutral**. Short comments like "Interesting approach" or "Good to know" carry a faint positive signal but no real enthusiasm — a reader cannot tell if the commenter is impressed or simply acknowledging the information. Similarly, "Finally someone made a video on this" reads as relief and praise, but could also signal frustration that the content was hard to find.

The second difficult boundary is **negative vs. neutral**. A question like "Why didn't you cover X?" is technically a question (neutral form) but implies the video was incomplete (negative content).

**Handling rule:** When a comment is ambiguous between two labels, label it by the dominant emotional signal. If no emotion is clearly dominant, default to neutral. Flag any comment where the decision took more than a few seconds of deliberation — these become the hard-cases list for a second-pass review after the full 200 examples are collected.

---

## 4. Data Collection Plan

**Sources:** Comment sections from 5–8 YouTube videos spanning different genres (educational tutorial, product review, music video, news/commentary, vlog) to ensure vocabulary diversity.

**Target:** 200 total examples, aiming for roughly 80 positive / 80 negative / 40 neutral. Neutral tends to be rarer in comment sections since most commenters write because they feel something, so a smaller quota is realistic.

**If a label is underrepresented after 200 examples:** Deliberately seek out videos where that sentiment is likely to dominate — for example, a controversial video for more negative comments, or a technical Q&A video for more neutral questions. Do not upsample by duplicating existing examples; collect fresh ones instead to keep the dataset varied.

---

## 5. Evaluation Metrics

Three metrics will be tracked:

- **Per-class F1 score** — The primary metric. Accuracy alone is misleading when classes are imbalanced (e.g., if 50% of comments are positive, a model that always predicts positive gets 50% accuracy while being useless). F1 balances precision and recall per label, making underperformance on minority classes visible.
- **Macro-averaged F1** — Averages F1 across all three labels without weighting by class size. This holds the model accountable to all three labels equally, which matters because neutral comments — the smallest class — are just as important to detect correctly as positive or negative ones.
- **Confusion matrix** — Not a single number, but essential for diagnosing *which* label pairs the model confuses most. If the model frequently mistakes negative for neutral, that is a different (and more serious) failure mode than confusing positive for neutral.

Accuracy is reported but not used as the primary decision metric.

---

## 6. Definition of Success

**Minimum threshold for deployment:** Macro-averaged F1 ≥ 0.75, with no individual label's F1 falling below 0.65. A classifier that hits those numbers is consistent enough to be useful in a real creator dashboard without systematically misrepresenting one category of audience response.

**"Good enough" bar:** A macro F1 of 0.80+ with a confusion matrix showing no single dominant failure pair. At that level the classifier is reliable enough to surface actionable signals — a creator can trust that a spike in "negative" comments reflects a genuine shift in audience reception, not label noise.

Below 0.75 macro F1, the classifier is not deployed; results would be presented as exploratory only.