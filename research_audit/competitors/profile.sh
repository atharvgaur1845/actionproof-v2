#!/bin/bash
# usage: profile.sh <repo_dir>  (read-only; never executes repo code)
cd "$1" || exit 1
echo "### HEAD"; git log -1 --format='%H %ci %an'
echo "### REMOTE BRANCHES"; git branch -r | grep -v HEAD | head -15
echo "### COMMITS (last 30)"; git log --format='%ci %s' -30 | cut -c1-140
echo "### FIRST COMMIT"; git log --reverse --format='%ci %s' | head -2 | cut -c1-120
echo "### TREE (dirs, depth2, file counts)"; find . -path ./.git -prune -o -type d -print | awk -F/ 'NF<=3' | while read d; do echo "$(find "$d" -maxdepth 1 -type f | wc -l) $d"; done | sort -k2 | head -60
echo "### EXT COUNTS"; git ls-files | sed -n 's/.*\.\([A-Za-z0-9]*\)$/\1/p' | sort | uniq -c | sort -rn | head -15
echo "### TOTAL FILES"; git ls-files | wc -l
echo "### TOP-LEVEL MD"; ls *.md 2>/dev/null; find . -path ./.git -prune -o -iname '*.md' -print | grep -viE 'node_modules' | head -60 | tr '\n' ' '; echo
echo "### RESULT-LIKE FILES"; git ls-files | grep -iE '(result|eval|metric|score|summary|report|log|trace)' | grep -iE '\.(json|jsonl|csv|tsv|txt|md|parquet)$' | head -50; echo "count: $(git ls-files | grep -iE '(result|eval|metric|score|summary|log|trace)' | grep -ciE '\.(json|jsonl|csv|parquet)$')"
echo "### HF / SPACE / VIDEO / SLIDES LINKS"; grep -rhoE 'https?://(huggingface\.co|hf\.co)/[A-Za-z0-9_./-]+|https?://[a-z0-9-]+\.hf\.space[^ )"]*|https?://(www\.)?(youtube\.com|youtu\.be|loom\.com|docs\.google\.com|drive\.google\.com|vercel\.app|[a-z0-9-]+\.vercel\.app|streamlit\.app|[a-z0-9-]+\.streamlit\.app)[^ )"]*' --include='*.md' --include='*.html' --include='*.py' --include='*.txt' --include='*.yaml' --include='*.yml' --include='*.json' . 2>/dev/null | sed 's/[.,]$//' | sort | uniq -c | sort -rn | head -40
echo "### COHERE MODELS"; grep -rhoiE '(CohereLabs|CohereForAI)/[A-Za-z0-9._-]+|tiny[- ]?aya[- a-z]*|aya[- ]expanse[- 0-9a-z]*|command[- ]r[a-z0-9+ -]*|aya[- ]vision[- 0-9a-z]*' --include='*.py' --include='*.md' --include='*.yaml' --include='*.yml' --include='*.json' --include='*.toml' . 2>/dev/null | tr 'A-Z' 'a-z' | sed 's/ *$//' | sort | uniq -c | sort -rn | head -12
echo "### TOOLKITS (py import/use count | md mention count)"; for t in agenttune auditkit curatorkit circuitkit aligntune lexsi; do echo "$t py:$(grep -rliE "$t" --include='*.py' . 2>/dev/null | wc -l) imports:$(grep -rhE "^(from|import) $t" --include='*.py' . 2>/dev/null | wc -l) md:$(grep -rliE "$t" --include='*.md' . 2>/dev/null | wc -l)"; done; grep -rhE '^(from|import) (agenttune|auditkit|curatorkit|circuitkit|aligntune)[a-z_.]*' --include='*.py' . 2>/dev/null | sort | uniq -c | sort -rn | head -8
echo "### TESTS"; find . -path ./.git -prune -o \( -name 'test_*.py' -o -name '*_test.py' \) -print | wc -l; echo "test fns: $(grep -rhE '^\s*def test_' --include='*.py' . 2>/dev/null | wc -l)"
echo "### CI"; ls .github/workflows 2>/dev/null
echo "### DEPS"; for f in requirements*.txt pyproject.toml uv.lock poetry.lock environment.yml package.json Dockerfile Makefile; do [ -e "$f" ] && echo "$f lines:$(wc -l <$f) pinned==:$(grep -c '==' $f)"; done
echo "### STATS KEYWORDS (files mentioning)"; for k in 'bootstrap' 'confidence interval|95% CI|\bCI\b' 'wilson' 'mcnemar' 'holm|bonferroni|benjamini' 'p-value|p_value|p =|p<' 'seed' 'preregist|pre-regist' 'held-out|heldout|held out' 'ablation' 'negative result|limitation' 'cohen|effect size' 'kappa|inter-annotator|human label'; do echo "$k: $(grep -rliE "$k" --include='*.md' --include='*.py' --include='*.json' . 2>/dev/null | wc -l)"; done
