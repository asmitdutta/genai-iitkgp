from transformers import pipeline
import pandas as pd

sentiment_analysis = pipeline("sentiment-analysis")

data = {
    "Employee": ["Amartya", "Sangeeta", "Faiz", "Khudiram"],
    "Feedback": [
        "i love the new project management tool",
        "the recent changes in the workflow are confusing",
        "great team collaboration in the last project",
        "i am not happy with the work environment"
    ]
}

df = pd.DataFrame(data)

def analyze_sentiment(feedback):
    result = sentiment_analysis(feedback)[0]
    return result["label"], result["score"]

df["Sentiment"], df["Confidence"] = zip(*df["Feedback"].apply(analyze_sentiment))
print(df)

positive_feedback = df[df["Sentiment"] == "POSITIVE"]
negative_feedback = df[df["Sentiment"] == "NEGATIVE"]

print(positive_feedback)
print(negative_feedback)