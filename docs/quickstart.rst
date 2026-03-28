Quickstart
==========

Install
-------

.. code-block:: bash

   pip install "prefgraph[datasets]"


.. code-block:: python

   from prefgraph.datasets import load_demo
   from prefgraph.engine import Engine

   users = load_demo()  # 100 synthetic consumers, no download
   engine = Engine(metrics=["garp", "ccei", "mpi", "hm"])
   results = engine.analyze_arrays(users)

   for r in results[:5]:
       print(r)


Load your own data
------------------

Use Polars for I/O and transformation, then hand off NumPy arrays or menu lists to the Engine.

Budget data from Parquet (wide format)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wide format means one row per observation with separate price and quantity columns for each good.

.. code-block:: python

   import polars as pl
   from prefgraph.engine import Engine

   # Example schema: user_id, t, p_milk, p_bread, q_milk, q_bread
   path = "my_budget_wide.parquet"
   df = pl.read_parquet(path)

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
   df = pl.read_parquet(path)

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
   menu_df = engine.analyze_menus(user_batches)
   print(menu_df[:3])
