#!/bin/sh
# vexp-hint: event-driven orientation hint (UserPromptSubmit). Fails open.
# vexp-engine: 1 - the engine through vexp's version-free link (~/.vexp/bin),
# else the newest vexp extension build under this home. Managed by vexp.
VEXP_BIN=
_vexp_exe=
case "$OSTYPE" in
  msys*|cygwin*) _vexp_exe=.exe ;;
  ?*) ;;
  *) case "$(uname -s 2>/dev/null)" in MINGW*|MSYS*|CYGWIN*) _vexp_exe=.exe ;; esac ;;
esac
for _vexp_base in "$VEXP_HOME" "$HOME" "$USERPROFILE"; do
  case "$_vexp_base" in /*|[A-Za-z]:[\\/]*) ;; *) continue ;; esac
  [ -f "$_vexp_base/.vexp/bin/vexp-core$_vexp_exe" ] && [ -x "$_vexp_base/.vexp/bin/vexp-core$_vexp_exe" ] || continue
  if [ -z "$_vexp_exe" ]; then
    [ -O "$_vexp_base/.vexp" ] || continue
    case "$(ls -ldL "$_vexp_base/.vexp" 2>/dev/null)" in
      ????????w*) continue ;;
      ?????w*) [ -G "$_vexp_base/.vexp" ] && [ "$(id -g)" = "$(id -u)" ] || continue ;;
    esac
  fi
  VEXP_BIN=""$_vexp_base/.vexp/bin/vexp-core$_vexp_exe
  break
done
if [ -z "$VEXP_BIN" ]; then
  VEXP_BIN=""$(for _vexp_c in "$HOME"/.vscode/extensions/vexp.vexp-vscode-*/binaries/*/vexp-core$_vexp_exe "$HOME"/.vscode-insiders/extensions/vexp.vexp-vscode-*/binaries/*/vexp-core$_vexp_exe "$HOME"/.vscode-oss/extensions/vexp.vexp-vscode-*/binaries/*/vexp-core$_vexp_exe "$HOME"/.vscode-server/extensions/vexp.vexp-vscode-*/binaries/*/vexp-core$_vexp_exe "$HOME"/.vscode-server-insiders/extensions/vexp.vexp-vscode-*/binaries/*/vexp-core$_vexp_exe "$HOME"/.vscode-remote/extensions/vexp.vexp-vscode-*/binaries/*/vexp-core$_vexp_exe "$HOME"/.cursor/extensions/vexp.vexp-vscode-*/binaries/*/vexp-core$_vexp_exe "$HOME"/.cursor-server/extensions/vexp.vexp-vscode-*/binaries/*/vexp-core$_vexp_exe "$HOME"/.windsurf/extensions/vexp.vexp-vscode-*/binaries/*/vexp-core$_vexp_exe "$HOME"/.windsurf-server/extensions/vexp.vexp-vscode-*/binaries/*/vexp-core$_vexp_exe "$HOME"/.trae/extensions/vexp.vexp-vscode-*/binaries/*/vexp-core$_vexp_exe "$HOME"/.trae-server/extensions/vexp.vexp-vscode-*/binaries/*/vexp-core$_vexp_exe "$HOME"/.antigravity/extensions/vexp.vexp-vscode-*/binaries/*/vexp-core$_vexp_exe "$HOME"/.antigravity-server/extensions/vexp.vexp-vscode-*/binaries/*/vexp-core$_vexp_exe "$HOME"/.kiro/extensions/vexp.vexp-vscode-*/binaries/*/vexp-core$_vexp_exe "$HOME"/.kiro-server/extensions/vexp.vexp-vscode-*/binaries/*/vexp-core$_vexp_exe; do [ -f "$_vexp_c" ] && printf '%s\n' "$_vexp_c"; done | sed -n 's|^.*/vexp\.vexp-vscode-\([^/ ]*\)/binaries/.*$|\1 &|p' | sort -V 2>/dev/null | tail -n 1 | sed 's/^[^ ]* //')
fi
[ -x "$VEXP_BIN" ] || exit 0
VEXP_HOOK_AGENT="claude-code" "$VEXP_BIN" prompt-hint 2>/dev/null
exit 0
