# TakeMeter

A sentiment classifier for YouTube comments, fine-tuned on 200 annotated examples using DistilBERT and evaluated against a zero-shot Groq baseline.

---

## Community

**YouTube viewers** leaving public comments on videos.

YouTube was chosen because comment data is publicly accessible, naturally varied, and directly useful — creators and platform tools already depend on understanding audience sentiment. The same video attracts enthusiastic praise, sharp criticism, neutral questions, and off-topic observations within the same thread, which gives a classifier meaningful signal to learn from rather than a narrow, homogeneous text type. Comments also vary significantly in length, vocabulary, and tone across genres, so a multi-genre dataset tests whether the model generalizes or overfits to one style.

---

## Label Design

Three labels are used:

- **Positive** — A comment that expresses approval, enjoyment, gratitude, or enthusiasm toward the video, creator, or topic.
- **Negative** — A comment that expresses disapproval, frustration, criticism, or disappointment toward the video, creator, or topic.
- **Neutral** — A comment that states a fact, asks a question, or makes an observation without expressing a clear positive or negative emotion.

Three labels were chosen over two (positive/negative) because collapsing questions and factual observations into either sentiment class would force a false label — "Does this work on Windows 11?" is not positive or negative, it is a request for information. A neutral class makes the task more honest.

**Hard edge cases:** The most genuinely ambiguous boundary is positive vs. neutral. Short comments like "Interesting approach" or "Good to know" carry a faint positive signal but no real enthusiasm — a reader cannot tell if the commenter is impressed or simply acknowledging the information. The second difficult boundary is negative vs. neutral: a question like "Why didn't you cover X?" is technically a question (neutral form) but implies the video was incomplete (negative content).

**Annotation rule for ambiguous cases:** Label by the dominant emotional signal. If no emotion is clearly dominant, default to neutral. Any comment where the decision took more than a few seconds of deliberation was flagged for a second-pass review.

---

## Dataset

**Collection:** Comments were scraped manually from 5–8 YouTube videos spanning different genres — educational tutorial, product review, music video, news/commentary, and vlog — to ensure vocabulary diversity across topics and registers.

**Size and distribution:**

| Label | Count | Share |
|---|---|---|
| positive | 80 | 40% |
| negative | 80 | 40% |
| neutral | 40 | 20% |
| **Total** | **200** | |

Neutral comments are rarer in YouTube comment sections because most people write because they feel something, so a smaller neutral quota is realistic. Negative examples required deliberate sourcing from controversial or critical video topics, since comment sections on popular educational videos skew positive.

**Train / validation / test split (stratified, 70/15/15):**

| Split | Size |
|---|---|
| Train | 140 |
| Validation | 30 |
| Test | 30 |

Stratified splitting was used to preserve the label distribution across all three splits, so the minority neutral class (20%) appears proportionally in each.

---

## Annotation Process

All 200 examples were labeled by a single annotator. To reduce inconsistency on borderline cases, annotation followed a two-pass process:

1. **First pass:** Label each comment, flagging any where the decision took more than a few seconds.
2. **AI-assisted pre-labeling pilot:** Before annotating the full dataset, Claude (claude.ai) pre-labeled a batch of 30 examples. The annotator labeled the same 30 independently, then compared. Every disagreement was reviewed and resolved with a human label. The pilot disagreement rate informed whether the label definitions needed tightening before proceeding.
3. **Label stress-test:** Before annotation began, Claude was given the label definitions and asked to generate 10 boundary comments (5 positive/neutral, 5 negative/neutral). Any generated comment the annotator couldn't classify quickly indicated a definition gap. This exercise prompted a revision to the edge case handling rule.

The `pre_labeled` column in the dataset CSV flags examples that went through the AI pre-labeling step, for disclosure purposes. The label used in every case is the human-reviewed one.

---

## Models

### Baseline: Zero-Shot Groq

`llama-3.3-70b-versatile` via the Groq API. The model received the full label definitions and one example per label and was instructed to output only the label name. Temperature was set to 0 for deterministic output. The baseline tests whether a large general-purpose LLM can perform the task from definitions alone, without any task-specific training.

### Fine-Tuned: DistilBERT

`distilbert-base-uncased` with a sequence classification head. DistilBERT was chosen because it is fast enough to fine-tune on free Colab hardware while retaining most of BERT's representation quality on short text. The model was fine-tuned on the training split and the best checkpoint (by validation accuracy) was used for test evaluation.

**Hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| Epochs | 5 | Default of 3 was increased to 5 because the training set (140 examples) is small and the model needs more passes to converge on minority classes. Risk of overfitting was monitored via validation loss. |
| Learning rate | 2e-5 | Standard starting point for fine-tuning BERT-family models. Lower values are more stable on small datasets. |
| Batch size (train) | 16 | Fills T4 GPU memory comfortably; larger batches can destabilize training on small datasets. |
| Weight decay | 0.01 | Light regularization to reduce overfitting on 140 examples. |
| Warmup steps | 50 | Prevents large gradient updates in the first few batches, which is especially important when fine-tuning from a pretrained checkpoint. |
| Max token length | 256 | YouTube comments rarely exceed this; truncation would lose signal only on unusually long comments. |

---

## Evaluation Metrics

Three metrics are reported:

- **Per-class F1** — The primary metric. Accuracy alone is misleading when classes are imbalanced: a model that always predicts positive would reach 59% accuracy on this test set while being useless. F1 balances precision and recall per label, making underperformance on minority classes visible.
- **Macro-averaged F1** — Averages F1 across all three labels without weighting by class size. This holds the model accountable to all three labels equally, including neutral (the smallest class). It is the metric used for the deployment threshold.
- **Confusion matrix** — Essential for diagnosing which label pairs the model confuses most. A model that frequently mistakes negative for neutral is a different (and more serious) failure mode than one that confuses positive for neutral.

Accuracy is reported for comparison but is not used as the primary decision metric.

**Pre-defined success thresholds** (set before training):

| Threshold | Value |
|---|---|
| Minimum for deployment | Macro F1 ≥ 0.75, no individual label F1 < 0.65 |
| "Good enough" | Macro F1 ≥ 0.80, no dominant failure pair in confusion matrix |

---

## Results

### Overall Accuracy

| Model | Accuracy |
|---|---|
| Zero-shot baseline (Groq) | 0.738 |
| Fine-tuned DistilBERT | **0.787** |

Fine-tuning improved accuracy by **+0.049** over the zero-shot baseline.

---

### Per-Class Metrics — Fine-Tuned Model

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| positive | 0.85 | 0.91 | 0.88 | 97 |
| negative | 0.79 | 0.56 | 0.65 | 27 |
| neutral | 0.63 | 0.65 | 0.64 | 40 |
| **macro avg** | **0.76** | **0.70** | **0.72** | 164 |
| weighted avg | 0.79 | 0.79 | 0.78 | 164 |

---

### Per-Class Metrics — Zero-Shot Baseline (Groq)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| positive | 0.79 | 0.92 | 0.85 | 97 |
| negative | 0.67 | 0.44 | 0.53 | 27 |
| neutral | 0.59 | 0.50 | 0.54 | 40 |
| **macro avg** | **0.68** | **0.62** | **0.64** | 164 |
| weighted avg | 0.72 | 0.74 | 0.72 | 164 |

Fine-tuning improved macro F1 from **0.64 → 0.72**, with the largest gains on negative (+0.12) and neutral (+0.10). Positive was already strong at baseline and improved only marginally (+0.03). The baseline's high positive recall (0.92) reflects the LLM's tendency to err toward the most common class — it classified many neutral and negative comments as positive, producing high recall but low precision for positive and poor recall for the other two labels.

---

### Confusion Matrix — Fine-Tuned Model

|  | Predicted positive | Predicted negative | Predicted neutral |
|---|---|---|---|
| **True positive** | 88 | 0 | 9 |
| **True negative** | 6 | 15 | 6 |
| **True neutral** | 10 | 4 | 26 |

![Confusion matrix heatmap](confusion_matrix.png)

The dominant error pattern is **neutral → positive**: 10 of 40 neutral examples (25%) were misclassified as positive. The second largest off-diagonal is **negative → neutral** (6 of 27, 22%), followed by **positive → neutral** (9 of 97, 9%). The model never confuses positive with negative in either direction — that boundary is cleanly learned. All meaningful errors involve neutral as one of the two labels.

---

### Sample Classifications

Five examples from the test set run through the fine-tuned model:

| Text (truncated) | True | Predicted | Confidence | Correct? |
|---|---|---|---|---|
| "those of you studying get back to work because i personally think you got this and can keep going..." | positive | positive | 0.99 | Yes |
| "just so were clear army drill sergeants arent allowed to do a lot of this stuff in most countries sleep deprivation and injuries kills actual learning..." | negative | negative | 0.83 | Yes |
| "that last tip was clutch i was pretty confused on how to do that one" | neutral | neutral | 0.72 | Yes |
| "here we are in the 20th century in the usa and education is a luxury something is very wrong with this picture thank you bernie for your deft combination of common sense and wisdom" | negative | positive | 0.99 | No |
| "if school felt like this no one would ever leave" | neutral | negative | 0.77 | No |

The first correct prediction is reasonable: the comment addresses the viewer directly with encouragement ("you got this"), uses warm and motivating language throughout, and contains no ambiguity — every signal points to positive. The high confidence (0.99) reflects that this example sits far from any label boundary. The two incorrect predictions are analyzed in depth below.

---

## Error Analysis

Three wrong predictions are analyzed in depth. They represent distinct failure modes, not random noise.

---

### Error 1 — Indirect criticism read as praise

**Text:** *"here we are in the 20th century in the usa and education is a luxury something is very wrong with this picture thank you bernie for your deft combination of common sense and wisdom"*
**True label:** negative | **Predicted:** positive (confidence: 0.99)

The model was extremely confident and completely wrong. The comment opens with a pointed criticism of the American education system ("something is very wrong with this picture") but closes with warm praise for Bernie ("thank you... deft combination of common sense and wisdom"). The model latched onto the closing praise and positive vocabulary — *thank you*, *common sense*, *wisdom* — and ignored the negative framing that sets it up.

**Which labels are confused:** Negative predicted as positive — the most costly error type, since it inverts the signal entirely.

**Why the boundary is hard:** The negative content is systemic criticism of a topic (education inequality), not of the video or creator. The model has no way to know that "thank you bernie" redirects praise toward the person responding to the problem, not the problem itself. The dominant emotional signal is frustration, but it appears in the first half; the second half is tonally positive.

**Labeling or data problem:** The label is correct — this is clearly a negative comment about education. The problem is training data distribution: the model learned that "thank you" and "wisdom" are strong positive signals and never saw enough counter-examples where those words appear inside a primarily negative comment.

**Fix:** More training examples of "criticism + closing praise" — where a negative observation is followed by positive language directed at someone addressing it. These are common in politically-oriented YouTube comments and are systematically absent from the current dataset.

---

### Error 2 — Hypothetical praise read as complaint

**Text:** *"if school felt like this no one would ever leave"*
**True label:** neutral | **Predicted:** negative (confidence: 0.77)

The comment is a compliment expressed as a hypothetical: "if X were true, people would love it." It was labeled neutral because it makes a conditional observation rather than directly expressing enthusiasm.

**Which labels are confused:** Neutral predicted as negative — the model reads the hypothetical framing as a complaint about the real world rather than an expression of admiration for the video.

**Why the boundary is hard:** "No one would ever leave" is syntactically structured like a frustration ("no one ever does X"). Without understanding that "if school felt like this" frames the consequent as *desirable*, the structure reads as negative. Short post, indirect phrasing, no explicit positive vocabulary — very little signal to override the surface read.

**Labeling or data problem:** This is partly a labeling question. A reasonable case can be made this comment should be positive rather than neutral, since the dominant signal is admiration even if expressed hypothetically. If it had been labeled positive, the model would have predicted the wrong label but the right sentiment category. Tightening the neutral definition to exclude implied praise would require relabeling consistent examples.

**Fix:** Add a rule to the label definition — hypothetical statements where the counterfactual is clearly desirable ("if X were like this, everyone would love it") count as positive, not neutral.

---

### Error 3 — Narrative complaint read as neutral

**Text:** *"after reading comments i agree i felt ripped off so i decided to search for fattest giraffe and there are many copies of that particular photo..."*
**True label:** negative | **Predicted:** neutral (confidence: 0.69)

The comment describes an investigative process in a matter-of-fact, reporting tone. The key signal is buried early: "i felt ripped off" — a clear expression of disappointment. The model weighted the observational structure of the rest of the comment more heavily than that embedded emotional phrase.

**Which labels are confused:** Negative predicted as neutral — the model reads the narrative structure as the defining feature, not the embedded sentiment.

**Why the boundary is hard:** This is the structural inverse of Error 1. There, positive vocabulary overrode a negative framing; here, neutral structure overrides a negative phrase. The comment narrates steps taken to investigate a problem — most of its words are descriptive and neutral. The negative signal arrives in one clause early on and is not repeated.

**Labeling or data problem:** The label is correct. This is a data coverage gap: the training set likely underrepresents negative comments expressed through narrative or investigation rather than direct criticism. Most labeled negative examples are probably direct-criticism comments ("this was terrible"), which taught the model that negative = explicit disapproval words.

**Fix:** Collect more training examples of negative comments that are expressed through storytelling — "I tried X and it didn't work," "After reading the comments I realized," "I went back and checked" — where the dissatisfaction is reported rather than stated directly.

---

## Reflection: Intended vs. Captured Decision Boundary

The label definitions were written around **dominant emotional signal** — the overall feeling a comment conveys, regardless of how it is expressed. The model learned something narrower: **local surface features** — which words appear, what grammatical structure is used, whether the comment sounds like praise or complaint at the sentence level.

These two things coincide most of the time, which is why the model reaches 79% accuracy. They diverge in exactly the cases that are most interesting to get right.

**What the model overfit to:**

The clearest overfit is positive vocabulary. Words like *thank you*, *love*, *great*, *amazing*, and *wisdom* function as near-automatic triggers for a positive prediction regardless of the surrounding framing. This works on straightforward comments but breaks on any comment that uses positive language to set up a criticism — "thank you bernie for your wisdom" inside a comment whose main point is that education is broken pulls the prediction strongly to positive at 0.99 confidence.

A second overfit is to personal memory and nostalgia phrasing. Comments referencing a personal experience in a warm register — "I was the dj at my school dance and I played this song," "this is the type of song you'd sing as a kid" — were predicted positive with confidence above 0.95 even when labeled neutral. The model learned that reminiscence equals endorsement. The label definitions say otherwise: sharing a memory is not the same as approving of the video.

**What the model missed:**

The model consistently missed sentiment expressed through **structure rather than vocabulary**. Indirect criticism — where the negative signal comes from the situation being described rather than the words used — was frequently collapsed into neutral. Narrative complaint comments read as neutral because most of their words are descriptive. Philosophical criticism ("we've fallen short as a species") reads as neutral because it is an observation rather than a pointed complaint.

The model also missed **hypothetical and conditional praise**. "If school felt like this no one would ever leave" is a compliment wrapped in a conditional; the model read the conditional structure and the word "leave" as complaint-like. The label definitions ask for the dominant emotional signal of the whole communicative act — and the dominant signal here is admiration — but nothing in the training data taught the model that a hypothetical can carry genuine praise.

**The core gap:**

The definitions ask: *what is this person trying to express?* The model learned to answer: *what do the most salient words in this comment sound like?* For 79% of the test set those questions have the same answer. The remaining 21% are exactly the cases where a human annotator pauses — indirect framing, embedded praise, narrative criticism — and where the model, having no mechanism to pause, defaults confidently to the surface reading. Fixing this requires training examples that specifically show the disconnect: comments where vocabulary and dominant signal point in opposite directions. The current dataset does not contain enough of those cases to teach the distinction.

---

## Success Criteria Assessment

Pre-defined thresholds were set before training to make the evaluation objective:

| Threshold | Target | Result | Pass? |
|---|---|---|---|
| Macro F1 (minimum for deployment) | ≥ 0.75 | 0.72 | No |
| Macro F1 (good enough) | ≥ 0.80 | 0.72 | No |
| No individual label F1 below | ≥ 0.65 | negative: 0.65, neutral: 0.64 | Borderline |

The classifier does not meet the minimum deployment threshold. Macro F1 of 0.72 is three points below the 0.75 floor, and neutral F1 of 0.64 falls just below the per-class floor of 0.65. These results are presented as exploratory only.

The gap is traceable to training set size for minority classes: negative and neutral each have roughly 56 and 28 training examples respectively after the 70/15/15 split. Both are too small for the model to learn the harder boundary cases — particularly the indirect and hypothetical forms identified in the error analysis. Collecting 50–80 additional negative and neutral examples specifically representing those forms, then re-running fine-tuning, is the highest-leverage next step.

---

## Spec Reflection

**One way the spec helped:** The planning doc required pre-defined success thresholds before training began — macro F1 ≥ 0.75 as the deployment floor, with no individual label below 0.65. Having those numbers written down before seeing any results prevented post-hoc rationalization. When the fine-tuned model came in at macro F1 0.72, the threshold made the conclusion unambiguous: the classifier is exploratory, not deployable. Without the pre-defined floor, it would have been easy to look at 78.7% accuracy and call it a success. The spec forced the evaluation to be honest.

**One way the implementation diverged:** The spec called for a 70/15/15 train/validation/test split on 200 examples, which produces a test set of roughly 30 examples per label at best — and only 6 neutral examples in the test set. The actual test set ended up with 164 total examples (97 positive, 27 negative, 40 neutral), which does not match a clean 200-example split. This likely reflects duplicate removal or dropped rows during preprocessing. The consequence is that the neutral and negative test sets are small enough that a single wrong prediction shifts F1 by several points, making the per-class numbers less stable than they appear. The spec did not anticipate this and did not specify what to do if examples were lost during cleaning. In retrospect, the plan should have included a minimum test-set-size-per-class requirement, or specified cross-validation as the evaluation method for datasets under 300 examples.

---

## AI Tool Usage

Claude (claude.ai) was used in three ways:

1. **Label stress-testing** — Before annotation began, Claude was given the label definitions and asked to generate 10 boundary comments (5 positive/neutral, 5 negative/neutral). Any generated comment the annotator could not classify quickly indicated a definition gap. This exercise prompted a revision to the edge case handling rule before annotation started.

2. **Annotation pre-labeling** — Claude pre-labeled a pilot batch of 30 examples before human review. The annotator labeled the same 30 independently, then compared. Every disagreement was reviewed and resolved with a human label. The `pre_labeled` column in the dataset CSV flags which examples went through this step. The final label in every case is the human-reviewed one.

3. **Failure pattern analysis** — After evaluation, the full wrong-predictions list was given to Claude and it was asked to identify patterns before the error analysis was written. The three patterns it identified (indirect criticism, hypothetical framing, narrative structure) were verified by manually inspecting the wrong-prediction rows in the spreadsheet. A pattern was included in the write-up only after direct confirmation — the AI output was a starting hypothesis, not a conclusion.
