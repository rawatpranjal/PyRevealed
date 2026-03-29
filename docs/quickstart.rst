Loading Data
============

Synthetic data (Rust-parallel generators)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PrefGraph ships four Rayon-parallel generators that produce data in the exact format the Engine expects. Each accepts a ``rationality`` parameter (0.0 = random, 1.0 = utility-maximizing) and a ``seed`` for reproducibility. Generation runs entirely in Rust with the GIL released, so 100K users take roughly one second.

.. code-block:: python

   from prefgraph import generate_random_budgets, generate_random_menus
   from prefgraph.engine import Engine, results_to_dataframe

   # Budget data: 100K users, 15 obs x 5 goods, Cobb-Douglas demand
   budget_data = generate_random_budgets(
       n_users=100_000, n_obs=15, n_goods=5,
       functional_form="cobb_douglas",   # also "ces" or "leontief"
       rationality=0.7, noise_scale=0.3, seed=42,
   )
   engine = Engine(metrics=["garp", "ccei", "hm"])
   df = results_to_dataframe(engine.analyze_arrays(budget_data))
   print(df[["is_garp", "ccei", "hm_consistent", "hm_total"]].head())

   # Menu data: 100K users, 10 obs, variable menu sizes 2-5
   menu_data = generate_random_menus(
       n_users=100_000, n_obs=10, n_items=5,
       menu_size=(2, 5), choice_model="logit",  # also "fixed_ranking" or "uniform"
       temperature=1.0, rationality=0.7, seed=42,
   )
   engine2 = Engine(metrics=["hm"])
   df2 = results_to_dataframe(engine2.analyze_menus(menu_data))
   print(df2[["is_sarp", "n_sarp_violations", "hm_consistent", "hm_total"]].head())

Production and intertemporal generators follow the same pattern:

.. code-block:: python

   from prefgraph import generate_random_production, generate_random_intertemporal

   # Production: 10K firms, 3 inputs + 2 outputs
   prod_data = generate_random_production(
       n_users=10_000, n_obs=15, n_inputs=3, n_outputs=2,
       functional_form="cobb_douglas", rationality=0.7, seed=42,
   )

   # Intertemporal: 10K agents, 5 time periods, discount factor 0.8-0.99
   inter_data = generate_random_intertemporal(
       n_users=10_000, n_obs=10, n_periods=5,
       discount_factor=(0.8, 0.99), rationality=0.7, seed=42,
   )

Both ``n_obs`` and ``menu_size`` accept an ``int`` for fixed counts or a ``(min, max)`` tuple for variable counts per user. A pure-NumPy fallback runs automatically if the Rust extension is unavailable.

Budget data from Parquet (wide format)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wide format means one row per observation with separate price and quantity columns for each good.

.. code-block:: python

   import polars as pl
   from prefgraph.engine import Engine

   # Example schema: user_id, t, p_milk, p_bread, q_milk, q_bread
   path = "my_budget_wide.parquet"

   engine = Engine(metrics=["garp", "ccei", "mpi", "hm"])  # batch, Rust-backed
   results_df = engine.analyze_parquet(
       path,
       user_col="user_id",
       cost_cols=["p_milk", "p_bread"],
       action_cols=["q_milk", "q_bread"],
   )
   print(results_df.head())

Budget data from Parquet (long format)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Long format means one row per (user, time, item) with columns for item id, price, and quantity.

.. code-block:: python

   import polars as pl
   from prefgraph.engine import Engine

   # Example schema: user_id, t, item, price, quantity
   path = "my_budget_long.parquet"

   engine = Engine(metrics=["garp", "ccei", "mpi", "hm"])  # batch, Rust-backed
   results_df = engine.analyze_parquet(
       path,
       user_col="user_id",
       item_col="item",
       time_col="t",
       cost_col="price",
       action_col="quantity",
   )
   print(results_df.head())

Budget data from a DataFrame (per‑user arrays)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have a DataFrame in memory, build per‑user price/quantity matrices and pass them to ``analyze_arrays``.

.. code-block:: python

   import polars as pl
   import numpy as np
   from prefgraph.engine import Engine

   # Long format: user_id, t, item, price, quantity
   df = pl.read_parquet("my_budget_long.parquet")

   users: list[tuple[np.ndarray, np.ndarray]] = []
   for uid, g in df.group_by("user_id", maintain_order=True):
       # Pivot items to columns ordered by item id for consistent matrices
       price_wide = g.pivot(values="price", index="t", on="item").sort("t").drop("t")
       qty_wide   = g.pivot(values="quantity", index="t", on="item").sort("t").drop("t")
       # Missing quantities imply zero; prices must be present
       P = price_wide.to_numpy()
       Q = qty_wide.fill_null(0).to_numpy()
       users.append((P, Q))

   engine = Engine(metrics=["garp", "ccei", "mpi", "hm"])  # batch, Rust-backed
   results = engine.analyze_arrays(users)
   print(results[0])

Menu data from Parquet (events → menus)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For clickstream data, build menus from what the user actually saw (e.g., viewed items) and use the purchased/clicked item as the choice. ``analyze_menus`` expects per‑user tuples ``(menus, choices, n_items)`` where menus are lists of item indices.

.. code-block:: python

   import polars as pl
   from prefgraph.engine import Engine

   # Example schema: user_id, session_id, event_type in {"view","purchase"}, product_id
   ev = pl.read_parquet("my_events.parquet").filter(
       pl.col("event_type").is_in(["view", "purchase"])  # keep only needed events
   )

   user_batches: list[tuple[list[list[int]], list[int], int]] = []

   for uid, ug in ev.group_by("user_id", maintain_order=True):
       # Build per-session menu and single choice
       views = ug.filter(pl.col("event_type") == "view").group_by("session_id").agg(
           pl.col("product_id").unique().alias("viewed")
       )
       buys = ug.filter(pl.col("event_type") == "purchase").group_by("session_id").agg(
           pl.col("product_id").n_unique().alias("n_buy"),
           pl.col("product_id").first().alias("choice")
       ).filter(pl.col("n_buy") == 1)

       sess = buys.join(views, on="session_id", how="inner")
       # Union viewed with purchased to guarantee choice ∈ menu; filter 2–50
       sess = sess.with_columns(
           pl.concat_list([pl.col("viewed"), pl.col("choice").map_elements(lambda x: [x])])
             .list.unique()
             .alias("menu")
       ).with_columns(pl.col("menu").list.len().alias("m"))
       sess = sess.filter((pl.col("m") >= 2) & (pl.col("m") <= 50))

       # Per‑user item remap to 0..N-1
       all_items = sorted({int(i) for ms in sess["menu"] for i in ms})
       to_local = {pid: i for i, pid in enumerate(all_items)}
       menus   = [[to_local[int(i)] for i in ms] for ms in sess["menu"]]
       choices = [to_local[int(c)] for c in sess["choice"]]
       n_items = len(all_items)

       if menus:
           user_batches.append((menus, choices, n_items))

   engine = Engine(metrics=["hm"])  # SARP/WARP/HM etc.
   results = engine.analyze_menus(user_batches)  # list[MenuResult]
   print(results[:3])
