#!/bin/bash

set -e
cd "$(git rev-parse --show-toplevel)"

echo "Renaming getiprompt -> instantlearn"
echo ""

PATTERNS=(
    "Geti Prompt:Geti Instant Learn"
    "GetiPrompt:InstantLearn"
    "GETI_PROMPT:INSTANT_LEARN"
    "GETIPROMPT:INSTANTLEARN"
    "geti-prompt:instant-learn"
    "geti_prompt:instant_learn"
    "getiprompt:instantlearn"
)

echo "Renaming directories:"
for pair in "${PATTERNS[@]}"; do
    old="${pair%%:*}"
    new="${pair##*:}"
    git ls-files | grep "$old" 2>/dev/null | xargs -I{} dirname {} 2>/dev/null | sort -u | \
        awk '{print length, $0}' | sort -rn | cut -d' ' -f2- | while read -r dir; do
        [[ "$(basename "$dir")" == *"$old"* ]] && [ -d "$dir" ] || continue
        newdir="${dir//$old/$new}"
        [ "$dir" != "$newdir" ] && git mv "$dir" "$newdir" && echo "  $dir -> $newdir"
    done
done


echo "Renaming Files:"
for pair in "${PATTERNS[@]}"; do
    old="${pair%%:*}"
    new="${pair##*:}"
    git ls-files 2>/dev/null | grep "$old" 2>/dev/null | while read -r file; do
        [[ "$(basename "$file")" == *"$old"* ]] && [ -f "$file" ] || continue
        newfile="${file//$old/$new}"
        [ "$file" != "$newfile" ] && git mv "$file" "$newfile" && echo "  $file -> $newfile"
    done
done

echo "Replacing Content:"
for pair in "${PATTERNS[@]}"; do
    old="${pair%%:*}"
    new="${pair##*:}"

    git grep -lI "$old" 2>/dev/null | grep -v "rename.sh" | while read -r file; do
        sed "s|$old|$new|g" "$file" > "$file.tmp" && mv "$file.tmp" "$file"
        echo "  $file: $old -> $new"
    done
done

git add -A
echo ""
echo "Done. Review with: git diff --staged"
