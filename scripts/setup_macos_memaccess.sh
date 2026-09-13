#!/usr/bin/env bash
# One-time macOS setup for live Dolphin RAM access (task_for_pid + mach_vm_remap).
#
# Prereqs (manual, once):  sudo DevToolsSecurity -enable   and user in the _developer group.
# Creates:
#   $DEST/Dolphin.app      copy of Dolphin re-signed with com.apple.security.get-task-allow
#   $VENV/bin/python-dbg   copy of the *real* interpreter signed with com.apple.security.cs.debugger
# The official /Applications/Dolphin.app and the Homebrew Python are not modified.
#
# Usage: scripts/setup_macos_memaccess.sh <venv_dir> [dest_dir]
# Note: dest must not be under ~/Documents if it is File Provider/iCloud-managed
#       (codesign fails with "resource fork, Finder information, or similar detritus").
set -euo pipefail
VENV="${1:?venv dir}"
DEST="${2:-$HOME/.cache/zelda-rl}"
SRC_APP="${DOLPHIN_APP:-/Applications/Dolphin.app}"
mkdir -p "$DEST"
TMP="$(mktemp -d)"

cat > "$TMP/dolphin.plist" <<'PL'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>com.apple.security.cs.allow-jit</key><true/>
<key>com.apple.security.cs.disable-library-validation</key><true/>
<key>com.apple.security.get-task-allow</key><true/>
</dict></plist>
PL
cat > "$TMP/python.plist" <<'PL'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>com.apple.security.cs.debugger</key><true/>
<key>com.apple.security.cs.disable-library-validation</key><true/>
<key>com.apple.security.cs.allow-unsigned-executable-memory</key><true/>
</dict></plist>
PL

rm -rf "$DEST/Dolphin.app"
ditto --norsrc --noextattr --noacl "$SRC_APP" "$DEST/Dolphin.app"
# Keep background/occluded training instances from being App-Napped (throttled/stalled).
/usr/libexec/PlistBuddy -c "Add :LSAppNapIsDisabled bool true" "$DEST/Dolphin.app/Contents/Info.plist" 2>/dev/null \
  || /usr/libexec/PlistBuddy -c "Set :LSAppNapIsDisabled true" "$DEST/Dolphin.app/Contents/Info.plist"
codesign --force --deep --sign - --options runtime --entitlements "$TMP/dolphin.plist" "$DEST/Dolphin.app"

# Framework Python's bin/python3.x re-execs Python.app/Contents/MacOS/Python, which would drop
# entitlements, so sign a copy of that inner binary. It finds the venv via ../pyvenv.cfg.
BASE="$("$VENV/bin/python" -c 'import sys; print(sys.base_prefix)')"
REAL="$BASE/Resources/Python.app/Contents/MacOS/Python"
[ -x "$REAL" ] || REAL="$(readlink -f "$VENV/bin/python")"
cp "$REAL" "$VENV/bin/python-dbg"
codesign --force --sign - --options runtime --entitlements "$TMP/python.plist" "$VENV/bin/python-dbg"

DevToolsSecurity -status
echo "export DOLPHIN_BIN=\"$DEST/Dolphin.app/Contents/MacOS/Dolphin\""
echo "run with: $VENV/bin/python-dbg <script>"
