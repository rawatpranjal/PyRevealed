# Dataset Choice-Structure Assessment

PrefGraph supports two consumer choice scenarios: **BehaviorLog** (budget choice — user
allocates quantities across goods given real prices) and **MenuChoiceLog** (discrete menu
choice — the same agent picks exactly one item from a known set, repeatedly over time).
This document assesses whether each benchmark dataset's choice structure genuinely fits one
of these scenarios. Implementation details, temporal ordering, and feature quality are out
of scope — the only question is whether the raw data looks like the scenario the package
models.

---

## Summary

| Dataset | Type | Verdict | Core issue |
|---------|------|---------|-----------|
| Dunnhumby | Budget | ✅ BehaviorLog | Real scanner prices, household quantities |
| Open E-Commerce | Budget | ✅ BehaviorLog | Real prices, persistent user IDs |
| H&M | Budget | ✅ BehaviorLog | Product-group goods, imputed prices, persistent customer IDs |
| FINN.no Slates | Menu | ✅ MenuChoiceLog | Platform-logged slates, single click, persistent users |
| MIND | Menu | ✅ MenuChoiceLog (sparse) | Platform-logged impression lists, 1-click filter; only ~5–15 obs/user |
| Instacart V2 (Menu) | Menu | ⚠️ MenuChoiceLog | Real choice (reorder SKU), but menu reconstructed from purchase history |
| REES46 | Menu | ⚠️ MenuChoiceLog | Real purchase as choice, but menu = self-selected session views |
| Taobao | Menu | ⚠️ MenuChoiceLog | Same as REES46 — session-reconstructed menu |
| Taobao (Buy Window) | Menu | ⚠️ MenuChoiceLog | Time-window reconstruction around purchase event |
| RetailRocket | Menu | ⚠️ MenuChoiceLog | Real purchase as choice, session-reconstructed menu, sparse users |
| Tenrec | Menu | ⚠️ MenuChoiceLog (weak) | Menu = clicks before first "like" — size set by feedback frequency, not presentation |
| KuaiRec | Menu | ❌ not MenuChoiceLog | User watches many videos; "choice" = argmax(watch\_ratio) assigned post-hoc |
| Yoochoose | Menu | ❌ not valid | No persistent user IDs; users are synthetic batches of anonymous sessions |

---

## Budget Datasets

### Dunnhumby

Household-level grocery scanner data from 2,500 households over two years. Each week a
household spends a total budget across 10 commodity categories at observed prices. This is a
textbook BehaviorLog setup: real prices, real expenditure quantities, persistent household
IDs, and a clear budget constraint. The scenario the package models directly.

### Open E-Commerce

Amazon purchase records from ~4,700 consumers across 50 product categories with observed
prices. Persistent user IDs, real monetary prices, and quantity decisions across repeated
purchase occasions. Fits BehaviorLog cleanly.

### H&M

Fashion transactions from 1.36M customers over two years. Product groups serve as goods;
prices are imputed via a three-tier fallback (own-period median → category median → grand
median) for periods with no purchase. Persistent customer IDs (first 12 chars). The price
imputation means some budget observations rest on inferred rather than observed prices, but
the scenario — allocating spend across garment categories — is a genuine budget problem.

---

## Menu Datasets

### FINN.no Slates

The strongest MenuChoiceLog fit in the suite. The platform directly logged every slate shown
to each user (up to 25 items per pageview) and which item was clicked. Menu = the logged
slate; choice = the single clicked item. User IDs are row indices in the npz array (persistent
within the dataset). No reconstruction is needed — the choice set was recorded by the
platform at the moment of decision.

### MIND

News impression logs from Microsoft News. Each row in `behaviors.tsv` is one impression
session: the platform showed the user a list of candidate articles (the menu) and recorded
which were clicked. Only impressions with exactly one click are retained, making the choice
unambiguous. User IDs (`U12345` format) are persistent across impressions. The match to
MenuChoiceLog is structurally correct. The practical limitation is data sparsity: MIND-small
covers one week, leaving most qualifying users with only 5–15 observations after the 1-click
filter.

### Instacart V2 (Menu)

Persistent user IDs from the original Instacart dataset. The choice is a real reorder
decision: the single SKU the user repurchased in a given (order, aisle) triplet. The menu is
reconstructed as all distinct products the user has purchased from that aisle in the trailing
three orders. This is a reasonable proxy for a consideration set — the user demonstrably knew
these products existed — but it is not an observed slate. The platform did not present these
items simultaneously; the analyst constructed the menu from purchase history.

### REES46

Multi-category e-commerce behavior (view, cart, purchase) from Oct–Nov 2019. Persistent
`user_id` values. The choice is a real purchase event (one item per session). The menu is
the set of items the user viewed in that session, assembled from click logs with a 30-minute
gap heuristic. The consideration set is self-selected by the user's own browsing, not
presented by the platform — a fundamental difference from FINN.no and MIND. It is the same
construction used by most recommendation benchmarks and is defensible as a weak-form menu,
but WARP/SARP tests will measure consistency of self-selected rather than platform-assigned
menus.

### Taobao

Taobao user behavior: 100M click/purchase events from ~1M users. Persistent `user_id`.
Choice = purchased item. Menu = items viewed in a gap-based session (30-minute inactivity
threshold) or in a trailing time window before the purchase (buy-window variant). Both
variants reconstruct the menu from click behavior. Same structural category as REES46.

### Taobao (Buy Window)

A variant of Taobao where the menu is defined as all items viewed within a fixed time window
immediately preceding each purchase. Identical structural issues to Taobao: persistent user
IDs, real choice, reconstructed menu.

### RetailRocket

Click-stream data from a large e-commerce platform with 1.4M visitors. Persistent
`visitorid`. Choice = the single item purchased in a session. Menu = all items the user
viewed in that session (plus the purchased item if not explicitly viewed). Same session-
reconstruction approach as REES46 and Taobao. The dataset has very few repeat purchasers
(minimum sessions lowered to 3 vs the standard 5), which means most qualifying users have a
thin preference graph.

### Tenrec

Tencent QQ Browser video data with persistent `user_id`. The choice is an item the user
"liked" (or followed/shared). The menu is defined as all items clicked since the previous
like event — a sliding window that resets on each feedback action. Two structural weaknesses
distinguish Tenrec from REES46/Taobao: (1) the menu size is entirely determined by how
frequently the user provides feedback, not by what was simultaneously presented; (2) the
choice is always the last item in the window by construction — the user did not select one
item over others that were available at the same time. The scenario approximates a menu
problem but the temporal structure of the consideration set is artificial.

### KuaiRec

KuaiRec records every video a user watched on a given day alongside `watch_ratio`
(play\_duration / video\_duration). User IDs are persistent. The menu is defined as all
videos watched in a day, and the "choice" is the video with the highest watch\_ratio.
This does not fit MenuChoiceLog. The user never made a single discrete choice — they watched
many videos. The "choice" is assigned post-hoc by the analyst from a continuous engagement
metric. The dataset is closer to a time-allocation problem (user allocates watching time
across a near-complete catalog) and would fit BehaviorLog semantics better than
MenuChoiceLog, if watch time were treated as the quantity and some price analog were
available.

### Yoochoose

RecSys 2015 click-stream dataset. Sessions are fully anonymous — there are no persistent
user IDs in the raw data. The loader works around this by grouping anonymous sessions by
item category and chunking them into synthetic users (`user_0`, `user_1`, ...). These
synthetic users are not real decision-makers; they are arbitrary batches of unrelated
shopping sessions from different people. RP analysis requires that repeated choices come from
the same agent. This requirement fails entirely. This dataset should not be used in the
benchmark.
