#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NON_GIT_EXIT_CODE="${REPO_SAFETY_NON_GIT_EXIT_CODE:-0}"
MAX_FILE_SIZE_MB="${REPO_SAFETY_MAX_FILE_MB:-50}"
MAX_FILE_SIZE_BYTES="$((MAX_FILE_SIZE_MB * 1024 * 1024))"

if ! command -v git >/dev/null 2>&1; then
  echo "[repo-safety] git is required."
  exit 2
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[repo-safety] not a git repository."
  echo "[repo-safety] set REPO_SAFETY_NON_GIT_EXIT_CODE=1 to fail in non-git directories."
  exit "$NON_GIT_EXIT_CODE"
fi

echo "[repo-safety] scanning tracked files..."

FAILURES=0
declare -a FAILURE_MESSAGES

add_failure() {
  local message="$1"
  FAILURE_MESSAGES+=("$message")
  FAILURES=$((FAILURES + 1))
}

file_size_bytes() {
  local path="$1"
  if stat -f%z "$path" >/dev/null 2>&1; then
    stat -f%z "$path"
    return
  fi
  stat -c%s "$path"
}

declare -a tracked_files
while IFS= read -r -d '' file; do
  tracked_files+=("$file")
done < <(git ls-files -z)

if [[ ${#tracked_files[@]} -eq 0 ]]; then
  echo "[repo-safety] no tracked files to scan."
  exit 0
fi

forbidden_file_hits=()
for file in "${tracked_files[@]}"; do
  case "$file" in
    *"/xcuserdata/"*|xcuserdata/*|*/DerivedData/*|DerivedData/*|*/.build/*|.build/*|*/build/*|build/*|*/dist/*|dist/*)
      forbidden_file_hits+=("$file")
      ;;
    *.p12|*.mobileprovision|*.provisionprofile|*.pem|*.key|*.cer|*.crt)
      forbidden_file_hits+=("$file")
      ;;
  esac
done
if [[ ${#forbidden_file_hits[@]} -gt 0 ]]; then
  add_failure "Forbidden tracked files detected:\n$(printf '  - %s\n' "${forbidden_file_hits[@]}")"
fi

oversized_files=()
for file in "${tracked_files[@]}"; do
  [[ -f "$file" ]] || continue
  size="$(file_size_bytes "$file")"
  if [[ "$size" -gt "$MAX_FILE_SIZE_BYTES" ]]; then
    size_mb="$(awk "BEGIN { printf \"%.2f\", $size / 1024 / 1024 }")"
    oversized_files+=("$file (${size_mb} MB)")
  fi
done
if [[ ${#oversized_files[@]} -gt 0 ]]; then
  add_failure "Large tracked files exceed ${MAX_FILE_SIZE_MB} MB:\n$(printf '  - %s\n' "${oversized_files[@]}")"
fi

declare -a content_checks=(
  "OpenAI API key|sk-[A-Za-z0-9]{20,}"
  "Google API key|AIza[0-9A-Za-z_-]{35}"
  "AWS access key|AKIA[0-9A-Z]{16}"
  "GitHub personal access token|gh[pousr]_[A-Za-z0-9_]{30,}"
  "GitHub fine-grained token|github_pat_[A-Za-z0-9_]{20,}"
  "Private key header|-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"
  "Certificate header|-----BEGIN CERTIFICATE-----"
  "Provisioning profile marker|<key>UUID</key>"
)

declare -a content_excludes=(
  ":(exclude)scripts/repo_safety_scan.sh"
  ":(exclude)docs/oss/security.md"
)

for rule in "${content_checks[@]}"; do
  rule_name="${rule%%|*}"
  rule_pattern="${rule#*|}"
  set +e
  matches="$(git grep -nI -E "$rule_pattern" -- . "${content_excludes[@]}" 2>/dev/null)"
  status=$?
  set -e
  if [[ $status -ne 1 && -n "${matches:-}" ]]; then
    add_failure "Potential secret pattern matched (${rule_name}):\n${matches}"
  fi
done

if [[ "$FAILURES" -gt 0 ]]; then
  echo "[repo-safety] FAILED with ${FAILURES} issue(s)."
  for message in "${FAILURE_MESSAGES[@]}"; do
    printf '\n%s\n' "$message"
  done
  cat <<'EOF'

Fix guidance:
  1. Remove sensitive files from git history/index: git rm --cached <file>
  2. Rotate any leaked keys immediately.
  3. Keep local secrets in Keychain, env vars, or git-ignored local files.
  4. For large files, use Git LFS or remove from tracked files.
EOF
  exit 1
fi

echo "[repo-safety] OK: no forbidden files, no oversized tracked files, no secret-like content hits."
