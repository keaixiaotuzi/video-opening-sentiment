# VADER vs Transformer: Does the Sentiment Method Change the Conclusion?

A robustness check for my master's thesis on the first 30 seconds of educational YouTube videos.

## Background

In my thesis, I studied how educational videos open. I took the transcript of the first 30 seconds of each video,
scored its sentiment with **VADER**, and compared high-view videos with low-view videos.

VADER is a word-list method. It gives each word a fixed score and adds them up. It is fast and easy to explain,
but it does not read context.

So I asked a simple question:

> If I score the same transcripts with a Transformer model that reads context, do my findings still hold?

## Method

```
Video transcripts (first 30 s)
        │
        ├── VADER (scores from the thesis)            → positive / neutral / negative
        │
        └── Twitter-RoBERTa (Transformer, Hugging Face) → positive / neutral / negative
                │
                ▼
1. How much do the two methods agree?   (agreement, Cohen's kappa, Spearman correlation)
2. Read the cases where they disagree
3. Re-run each thesis analysis with both methods and compare the results
```

- **Transformer model:** [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), used without fine-tuning
- **Labels:** standard VADER thresholds (±0.05); for the Transformer, the class with the highest probability
- **Hit vs regular:** the more-viewed and less-viewed groups of videos inside each channel
- **Statistical tests:** chi-square, Mann–Whitney U, Kruskal–Wallis
- **Tools:** Python, pandas, scipy, scikit-learn, Hugging Face Transformers, Google Colab (T4 GPU)

## What I found

**1. The two methods only partly agree.**
Agreement was moderate at best. Most disagreements were not "positive vs negative". They were about **neutral**:
VADER marked most openings as emotional, while the Transformer marked many of them as neutral.

**2. The reason is how each method reads text.**
Many educational videos start with a plain introduction. VADER sees words like *great*, *exciting* or *risk*
and gives a positive or negative score, even when the sentence is just describing a topic.
The Transformer reads the whole sentence. For example, in a sentence like *"This is going to blow your mind"*,
a word list can treat *blow* as negative, but a model that reads context understands it as excitement.

**3. One main finding is robust.**
The finding that **negative openings are more common in hit videos** held with both methods and was statistically significant
in both. Because two very different methods agree, this finding does not depend on the scoring tool.

**4. Some findings depend on the method or on how the data is grouped.**
- One channel-level finding changed a lot when the method changed.
- For video length, pooling all channels together gave the opposite direction to the channel-level results.
  This looks like **Simpson's paradox**: channels differ in both typical video length and opening style,
  so this kind of question should be tested **within each channel**.

**5. Data quality matters.**
Some transcripts came from speech recognition and machine translation, and a few were not readable.
Neither method can score those correctly, so they should be filtered out.

## Limitations

- **No ground truth yet.** The two methods disagree, but this project does not show which one is closer to human judgment.
- **Domain mismatch.** The Transformer was trained on tweets, not spoken video transcripts.
- **Coarser grouping.** "Hit vs regular" here is simpler than the levels used in the thesis.
- **Text only.** Tone of voice, music and visuals are lost when a video becomes a transcript.

## Next steps

1. Write a clear definition of a "negative opening" (negative topic, or negative tone?).
2. Label a random sample of openings by hand and use it as the test set to compare the methods.
3. Re-test the length finding within each channel.
4. If needed, fine-tune a model on the labelled data.

## How to run

1. Open `vader_vs_transformer.ipynb` in Google Colab.
2. Choose `Runtime → Change runtime type → T4 GPU`.
3. Choose `Runtime → Run all` and upload the two CSV files when asked.
   The columns each file needs are listed in Step 1 of the notebook.

## Data

The data is **not included** in this repository. The thesis paper is being prepared for publication,
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
This does not cover the thesis data, which is not shared here.
