# VADER vs Transformer: Does the Sentiment Method Change the Conclusion?

A robustness check for research that uses sentiment scores, with a reusable Colab notebook.

## Why this matters

Many studies score text with **VADER**, a word-list sentiment tool, and then build conclusions on those scores:
"negative headlines get more clicks", "positive reviews sell more", and so on.

VADER is fast and easy to explain, but it gives each word a fixed score and does not read context.
So a simple question is worth asking before trusting the conclusion:

> If the same texts are scored with a model that reads context, do the conclusions still hold?

This repository answers that question for one real case, and gives you a notebook to ask it about your own data.

## The case study

The example comes from a master's research project on the **first 30 seconds of educational YouTube videos**.
It compared the opening sentiment of high-view and low-view videos inside each channel.
The original analysis used VADER. Here, the same transcripts are scored again with
[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest),
and each original finding is tested again with both methods.

## Method

```
Texts (here: first 30 s of video transcripts)
        │
        ├── VADER (original scores)                   → positive / neutral / negative
        │
        └── Twitter-RoBERTa (Transformer, Hugging Face) → positive / neutral / negative
                │
                ▼
1. How much do the two methods agree?   (agreement, Cohen's kappa, Spearman correlation)
2. Read the cases where they disagree
3. Re-run each original analysis with both methods and compare the results
```

- **Transformer model:** used as it is, without fine-tuning
- **Labels:** standard VADER thresholds (±0.05); for the Transformer, the class with the highest probability
- **Statistical tests:** chi-square, Mann–Whitney U, Kruskal–Wallis
- **Tools:** Python, pandas, scipy, scikit-learn, Hugging Face Transformers, Google Colab (T4 GPU)

## Key findings

**1. The two methods only partly agree.**
Agreement was moderate at best. Most disagreements were not "positive vs negative". They were about **neutral**:
VADER marked most openings as emotional, while the Transformer marked many of them as neutral.

**2. The reason is how each method reads text.**
Many educational videos start with a plain introduction. VADER sees words like *great*, *exciting* or *risk*
and gives a positive or negative score, even when the sentence is just describing a topic.
The Transformer reads the whole sentence. For example, in *"This is going to blow your mind"*,
a word list can treat *blow* as negative, but a model that reads context understands it as excitement.

**3. Some conclusions are robust.**
The finding that **negative openings are more common in high-view videos** held with both methods and was statistically
significant in both. When two very different methods agree, the conclusion does not depend on the scoring tool.

**4. Other conclusions depend on the method or on how the data is grouped.**
- One channel-level finding changed a lot when the method changed.
- For video length, pooling all channels together gave the opposite direction to the channel-level results.
  This looks like **Simpson's paradox**: channels differ in both typical video length and opening style,
  so this kind of question should be tested **within each group**.

**5. Data quality matters.**
Some transcripts came from speech recognition and machine translation, and a few were not readable.
Neither method can score those correctly, so they should be filtered out first.

## Lessons for researchers using sentiment scores

1. **Check with a second method.** If a conclusion only appears with one tool, report it with care.
2. **Watch the "neutral" class.** Word-list tools often turn plain descriptions into positive or negative scores.
3. **Test inside groups before pooling.** Pooled results can point the opposite way (Simpson's paradox).
4. **Clean the text first.** Auto-generated or machine-translated text adds noise to any method.
5. **Get a small human-labelled sample.** It is the only way to know which method is closer to human judgment.

## Use it on your own data

The notebook works with any texts that already have VADER scores and some outcome to compare (views, sales, clicks…).

1. Open `vader_vs_transformer.ipynb` in Google Colab.
2. Choose `Runtime → Change runtime type → T4 GPU`.
3. Prepare your CSV files. Step 1 of the notebook lists the columns it expects
   (an `id`, the text, the original VADER score, and the outcome columns).
   If your column names are different, rename them in Step 1.
4. Choose `Runtime → Run all` and upload the files when asked.
5. Steps 2–4 (scoring and agreement) work for any text data.
   Steps 5–6 re-test the case-study findings; replace them with the comparisons from your own study.

## Limitations

- **No ground truth yet.** The two methods disagree, but this project does not show which one is closer to human judgment.
- **Domain mismatch.** The Transformer was trained on tweets, not spoken video transcripts.
- **Text only.** Tone of voice, music and visuals are lost when a video becomes a transcript.

## Next steps

1. Define clearly what counts as a "negative opening" (negative topic, or negative tone?).
2. Label a random sample by hand and use it as the test set to compare the methods.
3. Re-test the length finding within each channel.
4. If needed, fine-tune a model on the labelled data.

## Data

The case-study data is **not included**. The original research is being prepared for publication,
so this repository shares only the method and the overall findings.

## Files

```
├── README.md
├── vader_vs_transformer.ipynb   # the full analysis (no outputs saved)
├── .gitignore                   # keeps data and result files out of the repository
└── LICENSE                      # Apache License 2.0
```

## License

The code in this repository is released under the [Apache License 2.0](LICENSE).
This does not cover the case-study data, which is not shared here.
