from googleapiclient.discovery import build
import pandas as pd
import re
import json
from sklearn.linear_model import LinearRegression
import numpy as np

# ==========================================
# CONFIG
# ==========================================
key = API_KEY = "API_KEY"
CHANNEL_ID = "CHANNEL_ID"  # MrBeast

youtube = build("youtube", "v3", developerKey=API_KEY)


# ==========================================
# 1. DATA COLLECTION — 200 VIDEO (API)
# ==========================================
print("📥 Fetching videos from YouTube API...")

all_items = []
next_page = None

for i in range(4):  # 4 x 50 = 200 video
    request = youtube.search().list(
        part="snippet",
        channelId=CHANNEL_ID,
        maxResults=50,
        order="date",
        type="video",
        pageToken=next_page,
    )
    response = request.execute()
    all_items.extend(response["items"])
    next_page = response.get("nextPageToken")
    print(f"  Page {i+1}: {len(response['items'])} videos fetched")
    if not next_page:
        break

video_ids = [item["id"]["videoId"] for item in all_items]
titles = [item["snippet"]["title"] for item in all_items]
dates = [item["snippet"]["publishedAt"] for item in all_items]

print(f"✅ Total videos collected: {len(video_ids)}")


# ==========================================
# 2. GET STATISTICS (batch — 50 ta lik)
# ==========================================
print("\n📊 Fetching video statistics...")

views, likes, comments, durations = [], [], [], []

for i in range(0, len(video_ids), 50):
    batch = video_ids[i : i + 50]
    stats_response = (
        youtube.videos()
        .list(
            part="statistics,contentDetails",
            id=",".join(batch),
        )
        .execute()
    )

    for item in stats_response["items"]:
        stats = item["statistics"]
        views.append(int(stats.get("viewCount", 0)))
        likes.append(int(stats.get("likeCount", 0)))
        comments.append(int(stats.get("commentCount", 0)))
        durations.append(item["contentDetails"]["duration"])

print(f"✅ Statistics fetched for {len(views)} videos")


# ==========================================
# 3. DATA CLEANING & PROCESSING
# ==========================================
print("\n🔧 Processing data...")

df = pd.DataFrame(
    {
        "title": titles[: len(views)],
        "publish_date": dates[: len(views)],
        "views": views,
        "likes": likes,
        "comments": comments,
        "duration": durations,
    }
)


# Duration → seconds
def parse_duration(d):
    if not isinstance(d, str):
        return 0
    h = re.search(r"(\d+)H", d)
    m = re.search(r"(\d+)M", d)
    s = re.search(r"(\d+)S", d)
    total = 0
    if h:
        total += int(h.group(1)) * 3600
    if m:
        total += int(m.group(1)) * 60
    if s:
        total += int(s.group(1))
    return total


df["duration_sec"] = df["duration"].apply(parse_duration)
df["duration_min"] = (df["duration_sec"] / 60).round(1)


# Video type
def categorize(sec):
    if sec < 60:
        return "Shorts"
    if sec < 600:
        return "Medium"
    return "Long"


df["type"] = df["duration_sec"].apply(categorize)
df["publish_date"] = pd.to_datetime(df["publish_date"], utc=True)
df["publish_date"] = df["publish_date"].dt.tz_localize(None)
df["like_rate"] = (df["likes"] / df["views"] * 100).round(2)
df["comment_rate"] = (df["comments"] / df["views"] * 100).round(4)
df["month"] = df["publish_date"].dt.strftime("%Y-%m")

print(f"  Shorts: {(df['type']=='Shorts').sum()}")
print(f"  Long:   {(df['type']=='Long').sum()}")
print(f"  Medium: {(df['type']=='Medium').sum()}")


# ==========================================
# 4. TITLE KEYWORD ANALYSIS
# ==========================================
print("\n🔍 Keyword Analysis...")

keywords = [
    "$",
    "vs",
    "survive",
    "days",
    "win",
    "hours",
    "world",
    "last",
    "every",
    "free",
    "minutes",
    "challenge",
]

kw_results = []
for kw in keywords:
    mask = df["title"].str.lower().str.contains(kw.lower(), regex=False)
    count = int(mask.sum())
    avg_views = int(df[mask]["views"].mean()) if count > 0 else 0
    avg_likes = int(df[mask]["likes"].mean()) if count > 0 else 0
    kw_results.append(
        {
            "keyword": kw,
            "count": count,
            "avg_views": avg_views,
            "avg_likes": avg_likes,
        }
    )

kw_df = pd.DataFrame(kw_results).sort_values("avg_views", ascending=False)
print(kw_df.to_string(index=False))


# ==========================================
# 5. VIEWS PREDICTOR MODEL
# ==========================================
print("\n🤖 Training Views Predictor...")

df["is_short"] = (df["type"] == "Shorts").astype(int)
df["has_dollar"] = df["title"].str.contains(r"\$", regex=True).astype(int)
df["has_vs"] = df["title"].str.lower().str.contains("vs", regex=False).astype(int)
df["has_survive"] = (
    df["title"].str.lower().str.contains("survive", regex=False).astype(int)
)
df["has_win"] = df["title"].str.lower().str.contains("win", regex=False).astype(int)
df["has_days"] = df["title"].str.lower().str.contains("days", regex=False).astype(int)

features = ["is_short", "has_dollar", "has_vs", "has_survive", "has_win", "has_days"]
X = df[features]
y = df["views"]

model = LinearRegression()
model.fit(X, y)

score = model.score(X, y)
print(f"  Model R² score: {score:.3f}")

# Feature importance
importance = pd.DataFrame(
    {
        "feature": features,
        "impact": model.coef_.astype(int),
    }
).sort_values("impact", ascending=False)
print("\n  Feature Impact on Views:")
print(importance.to_string(index=False))

# Prediction examples
scenarios = [
    {"name": "Shorts + $ + win", "vals": [1, 1, 0, 0, 1, 0]},
    {"name": "Long + survive + days", "vals": [0, 0, 0, 1, 0, 1]},
    {"name": "Shorts + vs", "vals": [1, 0, 1, 0, 0, 0]},
]
print("\n  Prediction scenarios:")
for sc in scenarios:
    pred = int(model.predict([sc["vals"]])[0])
    print(f"  {sc['name']}: ~{pred:,} views")


# ==========================================
# 6. SUMMARY STATS
# ==========================================
print("\n📈 Channel Summary:")
print(f"  Total videos analyzed : {len(df)}")
print(f"  Total views           : {df['views'].sum():,}")
print(f"  Total likes           : {df['likes'].sum():,}")
print(f"  Avg views per video   : {int(df['views'].mean()):,}")
print(f"  Avg like rate         : {df['like_rate'].mean():.2f}%")
print(f"  Best video            : {df.loc[df['views'].idxmax(), 'title']}")
print(f"  Best video views      : {df['views'].max():,}")


# ==========================================
# 7. EXPORT
# ==========================================
print("\n💾 Saving files...")

# Excel — asosiy dataset
df.to_excel("mrbeast_videos.xlsx", index=False)
print("  ✅ mrbeast_videos.xlsx saved")

# Keyword analysis
kw_df.to_excel("keyword_analysis.xlsx", index=False)
print("  ✅ keyword_analysis.xlsx saved")

# JSON — dashboard uchun
export = {
    "kpis": {
        "total_videos": len(df),
        "total_views": int(df["views"].sum()),
        "total_likes": int(df["likes"].sum()),
        "avg_views": int(df["views"].mean()),
        "avg_like_rate": round(float(df["like_rate"].mean()), 2),
        "best_video": df.loc[df["views"].idxmax(), "title"],
        "best_views": int(df["views"].max()),
    },
    "type_stats": df.groupby("type")
    .agg(
        count=("title", "count"),
        avg_views=("views", "mean"),
        avg_likes=("likes", "mean"),
        avg_like_rate=("like_rate", "mean"),
    )
    .reset_index()
    .to_dict("records"),
    "keywords": kw_df.to_dict("records"),
    "top10": df.nlargest(10, "views")[
        ["title", "views", "likes", "like_rate", "type"]
    ].to_dict("records"),
}

with open("dashboard_data.json", "w", encoding="utf-8") as f:
    json.dump(export, f, ensure_ascii=False, indent=2)
print("  ✅ dashboard_data.json saved")

print("\n🎉 Analysis complete!")
