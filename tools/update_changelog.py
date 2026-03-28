import re
with open("CHANGELOG.md", "r") as f:
    text = f.read()
new_entry = "## [0.5.13] - 2026-03-28\n\n### Added\n- E-commerce benchmarks: consolidate Taobao (Buy-Anchored) results into main index and benchmarks.\n\n"
text = re.sub(r"(# Changelog\n\n)", r"\1" + new_entry, text)
with open("CHANGELOG.md", "w") as f:
    f.write(text)
