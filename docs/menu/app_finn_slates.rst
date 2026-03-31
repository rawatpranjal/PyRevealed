Classifieds Marketplace
=======================

Test whether users' click patterns on Norway's largest classifieds
marketplace reveal consistent category preferences, and examine what
drives inconsistency across search and recommendation interactions.

.. raw:: html

   <div style="margin: 1.5em 0;"></div>

.. image:: ../_static/fig2_consistency_portrait.png
   :width: 80%
   :align: center
   :alt: Distribution of Houtman-Maks consistency ratios across FINN.no users

.. raw:: html

   <div style="margin: 1.5em 0;"></div>

Platform and Data
-----------------

FINN.no is Norway's leading marketplace for real estate, vehicles, jobs,
and general merchandise. The platform recommends around 90 million items
each day, and roughly 20 percent of clicks come from its recommender
system. The FINN.no Slates dataset (Eide et al., RecSys 2021) records
37.5 million interactions from 2.3 million users over 30 days. Each
interaction logs every item shown in the slate and which item the user
clicked. About 70 percent of slates come from user-initiated search and
30 percent from recommendations.

Individual FINN.no listings are unique, so the same item rarely appears
in multiple slates. Analyzing preferences at the item level produces
very sparse preference graphs with only 6 percent item overlap. This
study instead maps each item to one of 290 category-geography groups
defined in the dataset metadata. Groups combine a product category with
a Norwegian county, such as "MOTOR, Rogaland" for motor vehicles in
Rogaland or "BAP, antiques, Trondlag" for antiques in the Trondlag
region. At the group level, overlap across slates rises to roughly 60
percent, which is dense enough for meaningful SARP and WARP analysis.

.. image:: ../_static/fig1_data_portrait.png
   :width: 90%
   :align: center
   :alt: Data portrait showing observations per user, groups per user, and top group frequencies

.. raw:: html

   <div style="margin: 1.5em 0;"></div>

The analysis loads the data using DuckDB for vectorized group mapping
and menu construction, then runs PrefGraph's batch Engine on the
resulting group-level MenuChoiceLogs.

.. code-block:: python

   from case_studies.finn_slates.group_loader import load_group_level
   from prefgraph import Engine

   user_logs, group_labels, stats = load_group_level(
       data_dir="/path/to/finn_slates",
       max_users=100_000,
   )

   engine = Engine()
   tuples = [log.to_engine_tuple() for log in user_logs.values()]
   results = engine.analyze_menus(tuples)


How Consistent Are Classifieds Users?
-------------------------------------

Running SARP, WARP, and Houtman-Maks analysis on the full population
reveals that most classifieds users do not maintain a perfectly
consistent ranking of category-geography groups. The SARP pass rate is
around 12 percent, meaning only about one in eight users makes click
choices that are fully consistent with a stable preference ordering over
item groups. The remaining 88 percent have at least one preference cycle
in their choice history.

The Houtman-Maks consistency ratio provides a more graded picture. The
mean HM ratio across all users is approximately 0.88, indicating that
roughly 88 percent of each user's choices can be explained by a single
preference ranking. The median is similar. This is substantially higher
than a random-choice baseline, where the SARP pass rate drops to about
6 percent. Real users are more consistent than random, but far from
perfectly rational.

.. image:: ../_static/fig2_consistency_portrait.png
   :width: 80%
   :align: center
   :alt: Consistency portrait with HM ratio distribution and SARP violation counts

.. raw:: html

   <div style="margin: 1.5em 0;"></div>


Does Stochastic Choice Tell a Different Story?
-----------------------------------------------

Stochastic choice analysis requires repeated presentations of the same
menu to build choice frequency distributions. At the group level, most
users still see distinct menus at each interaction because the
combination of groups in a slate varies across sessions. Only a small
fraction of users have three or more repeated group-level menus, which
is the minimum needed for a meaningful Random Utility Model test. This
sparsity is itself a finding. Even after collapsing 1.3 million unique
items into 290 groups, the combinatorial diversity of slate compositions
on a large marketplace limits the applicability of stochastic choice
models.

.. image:: ../_static/fig3_stochastic.png
   :width: 55%
   :align: center
   :alt: Distribution of repeated menus per user

.. raw:: html

   <div style="margin: 1.5em 0;"></div>


Search Intent and Preference Clarity
-------------------------------------

The dataset distinguishes between search-initiated slates (the user
typed a query) and recommendation-initiated slates (the platform
surfaced items). Users who predominantly click from search results might
be expected to show more consistent preferences because they have a
clearer intent. Splitting users into search-heavy and
recommendation-heavy terciles and comparing their HM ratios tests this
hypothesis.

.. image:: ../_static/fig4_search_vs_reco.png
   :width: 55%
   :align: center
   :alt: Overlapping distributions of HM ratio for search-heavy vs recommendation-heavy users

.. raw:: html

   <div style="margin: 1.5em 0;"></div>


What Drives Inconsistency?
--------------------------

For users who fail SARP, the structure of their violations reveals
whether inconsistency comes from borderline indifference between similar
groups or from genuinely confused preferences. The distribution of
strongly connected component sizes in the preference graph shows how
tangled each user's preference cycles are. Most violators have small
SCCs of two to five groups, suggesting that violations typically involve
a handful of confusable category-geography pairs rather than wholesale
randomness.

.. image:: ../_static/fig5_violations.png
   :width: 80%
   :align: center
   :alt: SCC size distribution and cycle length distribution for SARP violators

.. raw:: html

   <div style="margin: 1.5em 0;"></div>


Pipeline and Code
-----------------

The analysis script at ``case_studies/finn_slates/run_analysis.py``
reproduces all results and figures. It requires the FINN.no Slates
dataset (Eide et al., 2021) available from the dataset repository at
``github.com/finn-no/recsys-slates-dataset``. Three files are needed:
``data.npz`` (the main interaction data), ``itemattr.npz`` (item-to-group
mapping), and ``ind2val.json`` (human-readable group labels).

.. code-block:: bash

   python3 case_studies/finn_slates/run_analysis.py \
       --data-dir /path/to/finn_slates \
       --max-users 100000
