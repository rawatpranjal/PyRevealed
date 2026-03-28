# Tenrec Analysis Report: Facts Only

**Dataset:** QB-video.csv sample (500,000 rows)
**Users:** 3,120
**Items:** 81,547
**Sample load:** `pl.read_csv(csv_path, null_values='\\N')`

---

## Part 1: EDA Diagnostics (6 tests)

### Diagnostic 1: Action-State Consistency

**Query:** Count rows by (click, like, share, follow) combination

| Action Type | Count | % |
|---|---:|---:|
| click=0, like=0, share=0, follow=0 | 279,705 | 55.9% |
| click=1, like=0, share=0, follow=0 | 211,319 | 42.3% |
| click=1, like=1, share=0, follow=0 | 5,644 | 1.1% |
| click=0, like=1, share=0, follow=0 | 2,368 | 0.5% |
| click=1, like=0, share=1, follow=0 | 238 | 0.0% |
| click=1, like=0, share=0, follow=1 | 232 | 0.0% |
| click=0, like=0, share=0, follow=1 | 149 | 0.0% |
| click=1, like=1, share=0, follow=1 | 123 | 0.0% |
| click=0, like=0, share=1, follow=0 | 70 | 0.0% |
| click=0, like=1, share=0, follow=1 | 62 | 0.0% |

**Derived counts:**
- Pure exposures: 279,705 (55.9%)
- Like without click: 2,450 (0.5%)
- Share without click: 90 (0.0%)
- Follow without click: 219 (0.0%)

---

### Diagnostic 2: Window-Size Distribution (5 Rules)

**Rule 1: Exposures since last like**

Query: For each user, accumulate all rows until a like event.

| Metric | Value |
|---|---:|
| Median | 2.0 |
| p75 | 5.0 |
| p90 | 47.0 |
| p95 | 142.1 |
| p99 | 372.3 |
| % size=1 | 37.9% |
| % > 20 | 17.1% |
| % > 50 | 9.6% |
| Chosen-last share | 100.0% |
| Top-category share | 83.5% |
| Number of windows | 8,092 |

**Rule 2: Clicks since last like**

Query: Accumulate only clicked items until a like event.

| Metric | Value |
|---|---:|
| Median | 2.0 |
| p75 | 3.0 |
| p90 | 26.0 |
| p95 | 67.4 |
| p99 | 198.7 |
| % size=1 | 45.6% |
| % > 20 | 11.9% |
| % > 50 | 6.0% |
| Chosen-last share | 100.0% |
| Top-category share | 85.9% |
| Number of windows | 6,833 |

**Rule 3: Exposures since last positive action (like OR share OR follow)**

Query: Accumulate all rows until any like/share/follow event.

| Metric | Value |
|---|---:|
| Median | 3.0 |
| p75 | 7.0 |
| p90 | 57.0 |
| p95 | 165.5 |
| p99 | 360.2 |
| % size=1 | 35.7% |
| % > 20 | 19.3% |
| % > 50 | 10.9% |
| Chosen-last share | 100.0% |
| Top-category share | 82.7% |
| Number of windows | 8,344 |

**Rule 4: Last 5 exposures before a like**

Query: For each like event, use rows [i-4:i+1] where i is the like row.

| Metric | Value |
|---|---:|
| Median | 5.0 |
| p75 | 5.0 |
| p90 | 5.0 |
| p95 | 5.0 |
| p99 | 5.0 |
| % size=1 | 0.4% |
| % > 20 | 0.0% |
| % > 50 | 0.0% |
| Chosen-last share | 100.0% |
| Top-category share | 75.6% |
| Number of windows | 8,092 |

**Rule 5: Last 10 exposures before a like**

Query: For each like event, use rows [i-9:i+1] where i is the like row.

| Metric | Value |
|---|---:|
| Median | 10.0 |
| p75 | 10.0 |
| p90 | 10.0 |
| p95 | 10.0 |
| p99 | 10.0 |
| % size=1 | 0.4% |
| % > 20 | 0.0% |
| % > 50 | 0.0% |
| Chosen-last share | 100.0% |
| Top-category share | 71.5% |
| Number of windows | 8,092 |

---

### Diagnostic 3: Category Coherence (Rule 2)

**Query: Unique categories per window**

```python
counter = Counter(window_cats)
unique_counts.append(len(set(window_cats)))
```

| Metric | Value |
|---|---:|
| Median | 1.0 |
| p75 | 1.0 |
| p90 | 2.0 |
| p95 | 2.0 |
| p99 | 2.0 |

**Query: Category entropy per window**

```python
probs = np.array(list(counter.values())) / len(window_cats)
ent = -np.sum(probs * np.log(probs + 1e-10))
```

| Metric | Value |
|---|---:|
| Median | 0.0 |
| p90 | 0.69 |
| p99 | 0.69 |

**Query: Top-category share**

```python
top_count = max(counter.values())
top_share = top_count / len(window_cats)
```

| Metric | Value |
|---|---:|
| Median | 1.00 |
| p90 | 1.00 |
| p99 | 1.00 |

---

### Diagnostic 4: Position of Chosen Item (Rule 2)

**Query: Rank of liked item within click-window**

```python
for row in user_df.iter_rows(named=True):
    if row["click"] == 1:
        window.append(row["item_id"])
        like_position = len(window)
    if row["like"] == 1:
        positions.append(like_position)
        normalized_positions.append(like_position / len(window))
```

| Metric | Value |
|---|---:|
| Median rank | 2.0 |
| Median normalized position | 1.00 |
| % always last (normalized = 1.0) | 100.0% |

**Position distribution:**

| Bucket | Count | % |
|---|---:|---:|
| [0.0–0.2) | 0 | 0.0% |
| [0.2–0.4) | 0 | 0.0% |
| [0.4–0.6) | 0 | 0.0% |
| [0.6–0.8) | 0 | 0.0% |
| [0.8–1.0] | 6,833 | 100.0% |

---

### Diagnostic 5: Watch-Time Separation

**Query 5.1: Liked items**

```python
liked = df.filter(pl.col("like") == 1)
liked_times = liked["watching_times"].to_list()
```

| Metric | Value |
|---|---:|
| Median | 2.0 |
| Mean | 2.5 |
| Count | 8,285 |

**Query 5.2: Clicked but not liked**

```python
clicked_not_liked = df.filter((pl.col("click") == 1) & (pl.col("like") == 0))
```

| Metric | Value |
|---|---:|
| Median | 2.0 |
| Mean | 2.4 |
| Count | 211,791 |

**Query 5.3: Exposed only (no click, no like)**

```python
exposed = df.filter((pl.col("click") == 0) & (pl.col("like") == 0))
```

| Metric | Value |
|---|---:|
| Median | 1.0 |
| Mean | 0.9 |
| Count | 279,924 |

---

### Diagnostic 6: User-Level Pathology Rates

**Query 6.1: Per-user like rate**

```python
likes = (user_df["like"] == 1).sum()
like_rate = likes / n_rows if n_rows else 0
like_rates.append(like_rate)
```

| Metric | Value |
|---|---:|
| Median | 0.000 |
| p90 | 0.027 |
| p99 | 0.265 |

**Query 6.2: Per-user positive-action rate**

```python
positives = ((user_df["click"] == 1) | (user_df["like"] == 1) |
             (user_df["share"] == 1) | (user_df["follow"] == 1)).sum()
pos_rate = positives / n_rows if n_rows else 0
```

| Metric | Value |
|---|---:|
| Median | 0.429 |
| p90 | 0.731 |
| p99 | 1.000 |

**Query 6.3: Per-user size-1 window share (Rule 2)**

```python
window_sizes = []
for row in user_df.iter_rows(named=True):
    if row["click"] == 1:
        window.append(row["item_id"])
    if row["like"] == 1:
        window_sizes.append(len(window))
size_1_windows = sum(1 for s in window_sizes if s == 1)
size_1_share = size_1_windows / len(window_sizes) if window_sizes else 0
```

| Metric | Value |
|---|---:|
| Median | 0.000 |
| p90 | 0.000 |
| p99 | 0.667 |
| Users with > 50% size-1 windows | 59 (1.9%) |

**Query 6.4: Per-user median window size (Rule 2)**

```python
median_window_size = np.median(window_sizes) if window_sizes else 0
median_windows.append(median_window_size)
```

| Metric | Value |
|---|---:|
| Median across users | 0.0 |
| p90 across users | 21.0 |

---

## Part 2: Salvage Experiment (3 constructions)

### Construction A: Category-Run Micro-Sessions

**Query:** Build sessions as contiguous rows in same category. Reset on category switch or positive action.

```python
for user_id in df_sorted["user_id"].unique():
    user_df = df_sorted.filter(pl.col("user_id") == user_id)
    session = []
    last_cat = None
    for row in user_df.iter_rows(named=True):
        cat = row["video_category"]
        if last_cat is not None and cat != last_cat:
            sessions.append({"items": session, ...})
            session = []
        session.append(row["item_id"])
        if row["like"] == 1 or row["share"] == 1 or row["follow"] == 1:
            sessions.append({"items": session, ...})
            session = []
        last_cat = cat
```

**Results:**

| Metric | Value |
|---|---:|
| Number of sessions | 211,684 |
| Median session size | 1.0 |
| p95 session size | 6.0 |
| p99 session size | 12.0 |
| % sessions with duplicates | 0.0% |
| % sessions with same category | 99.8% |
| % sessions with chosen-item-last | 100.0% |

---

### Construction B1: Fixed K=5 Window Before Like

**Query:** For each like event, take rows [i-4:i+1].

```python
for i, row in enumerate(user_df.iter_rows(named=True)):
    if row["like"] == 1:
        start_idx = max(0, i - 4)
        window_rows = user_df[start_idx:i+1]
        windows.append({
            "items": window_rows["item_id"].to_list(),
            "categories": window_rows["video_category"].to_list(),
            "size": len(window_rows),
        })
```

**Results:**

| Metric | Value |
|---|---:|
| Number of windows | 8,285 |
| Median window size | 5.0 |
| p95 window size | 5.0 |
| p99 window size | 5.0 |
| % windows with duplicates | 0.0% |
| % windows with same category | 22.1% |
| % windows with chosen-item-last | 100.0% |

---

### Construction B2: Fixed K=10 Window Before Like

**Query:** For each like event, take rows [i-9:i+1].

```python
for i, row in enumerate(user_df.iter_rows(named=True)):
    if row["like"] == 1:
        start_idx = max(0, i - 9)
        window_rows = user_df[start_idx:i+1]
        windows.append({
            "items": window_rows["item_id"].to_list(),
            "categories": window_rows["video_category"].to_list(),
            "size": len(window_rows),
        })
```

**Results:**

| Metric | Value |
|---|---:|
| Number of windows | 8,285 |
| Median window size | 10.0 |
| p95 window size | 10.0 |
| p99 window size | 10.0 |
| % windows with duplicates | 0.0% |
| % windows with same category | 8.6% |
| % windows with chosen-item-last | 100.0% |

---

## Acceptance Criteria Testing

**Criteria code:**

```python
criteria = {
    "Median unique items >= 4": lambda r: r["median_size"] >= 4,
    "Duplicate share < 15%": lambda r: r["pct_duplicates"] < 15,
    "Top-category share > 70%": lambda r: r["pct_same_category"] > 70,
    "Chosen-last share < 100%": lambda r: r["chosen_last_share"] < 100,
    "N sessions > 100": lambda r: r["n_sessions"] > 100,
}
```

**Results:**

| Criterion | Cat-Run | K=5 | K=10 | Pass Count |
|---|---|---|---|---|
| Median ≥ 4 | ✗ 1.0 | ✓ 5.0 | ✓ 10.0 | 2/3 |
| Dup% < 15 | ✓ 0.0% | ✓ 0.0% | ✓ 0.0% | 3/3 |
| Same-cat% > 70 | ✓ 99.8% | ✗ 22.1% | ✗ 8.6% | 1/3 |
| Chosen-last < 100 | ✗ 100.0% | ✗ 100.0% | ✗ 100.0% | 0/3 |
| N > 100 | ✓ 211K | ✓ 8.3K | ✓ 8.3K | 3/3 |
| **Total Passing** | **3/5** | **3/5** | **3/5** | **9/15** |

**Threshold:** 4/5 (80%)

| Construction | Passing | Threshold | Status |
|---|---:|---|---|
| Cat-run | 3/5 | 4/5 | ✗ FAIL |
| K=5 | 3/5 | 4/5 | ✗ FAIL |
| K=10 | 3/5 | 4/5 | ✗ FAIL |

---

## Summary Table: EDA + Salvage Combined

| Metric | D2-R1 | D2-R2 | D2-R3 | D2-R4 | D2-R5 | Cat-Run | K=5 | K=10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Median size | 2.0 | 2.0 | 3.0 | 5.0 | 10.0 | 1.0 | 5.0 | 10.0 |
| p99 size | 372.3 | 198.7 | 360.2 | 5.0 | 10.0 | 12.0 | 5.0 | 10.0 |
| % size=1 | 37.9% | 45.6% | 35.7% | 0.4% | 0.4% | — | — | — |
| Same-cat% | 83.5% | 85.9% | 82.7% | 75.6% | 71.5% | 99.8% | 22.1% | 8.6% |
| Chosen-last% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Dup% | — | — | — | — | — | 0.0% | 0.0% | 0.0% |
| N windows | 8,092 | 6,833 | 8,344 | 8,092 | 8,092 | 211,684 | 8,285 | 8,285 |

---

## Data Files

**Input:** `~/.prefgraph/data/tenrec/QB-video.csv` (2.44M rows, 74MB)
**Sample:** First 500K rows extracted
**Load:** `pl.read_csv(csv_path, null_values='\\N')`

**Scripts:**
- `examples/eda/tenrec_eda.py`
- `examples/eda/tenrec_salvage.py`

**Columns:** user_id, item_id, click, follow, like, share, video_category, watching_times, gender, age
