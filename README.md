# FLAO - FiveM Lua Auto Optimizer

AST-based Lua analyzer and optimizer for FiveM/GTA 5 resources.
Built for **Lua 5.3/5.4** (FiveM runtime) with LuaJIT compatibility.
Experimental but battle-tested with large resource collections.

Originally based on [ALAO](https://github.com/Anomaly-ALAO/ALAO) by Abraham (Priler).

## How it works

Lua code is parsed into an AST (abstract syntax tree), which allows safe code manipulation without breaking things. FLAO analyzes the AST to find performance issues common in FiveM scripts and can automatically fix many of them.

Key optimizations include:
- Caching expensive native calls like `PlayerPedId()`, `GetEntityCoords()`, etc.
- Replacing `GetDistanceBetweenCoords()` suggestions with vector math `#(v1 - v2)`
- Fixing O(n²) string concatenation in loops
- Replacing deprecated Lua patterns with faster alternatives

## Quick Start

```bash
python fivem_lua_lint.py [path_to_resources] [options]

# Basic Usage (combinable)
--fix              Fix safe (GREEN) issues automatically
--fix-yellow       Fix unsafe (YELLOW) issues automatically
--fix-debug        Comment out debug statements (log, printf, print, etc.)
--experimental     Enable experimental fixes (string concat in loops, branch-aware counting)
--cache-threshold N  Minimum call count to trigger caching (default: 4)

--direct           Process scripts directly (single .lua file or folder, no resource structure)
--exclude "exclude.txt"  Exclude certain resources from reports/fixes

# Experimental features
--fix-nil          Auto-fix some nil checks (that could cause errors)
--remove-dead-code / --debloat  Remove dead code from scripts

# Reports & Restore
--report [file]    Generate comprehensive report (.txt, .html, .json)
--revert           Restore all .flao-bak backup files (undo fixes)

# Performance
--timeout [seconds]  Timeout per file (default: 10)
--workers / -j       Parallel workers for fixes (default: CPU count)

# Output
--verbose / -v     Show detailed output
--quiet / -q       Only show summary

# Backup Management
--backup / --no-backup        Create .flao-bak files before modifying (default: True)
--list-backups                List all .flao-bak backup files
--backup-all-scripts          Backup ALL scripts to a zip archive before modifications

# Danger Zone
--clean-backups    Remove all .flao-bak backup files
```

## Requirements

```
- Python 3.8+
- luaparser
- jinja2
```

Install dependencies:
```bash
pip install luaparser jinja2
```

## Currently Detected Patterns

### GREEN (safe to auto-fix)
- `table.insert(t, v)` → `t[#t+1] = v`
- `table.getn(t)` → `#t`
- `string.len(s)` → `#s`
- `math.pow(x, 2)` → `x*x`
- `math.pow(x, 0.5)` → `math.sqrt(x)`
- Uncached native calls used 3+ times:
  - `PlayerPedId()`, `PlayerId()`, `GetPlayerServerId()`
  - `GetEntityCoords()`, `GetEntityModel()`, `GetEntityHeading()`
  - `GetHashKey()`, `GetPlayerPed()`, `GetVehiclePedIsIn()`

### YELLOW (review needed)
- `GetDistanceBetweenCoords()` → use `#(coords1 - coords2)` vector math
- String concatenation in loops (`s = s .. x`) - fixable with `--experimental`
- Potential nil access on natives that can return 0/nil

### RED (info only, no auto-fix)
- Global variable writes

### DEBUG (can be auto commented out)
- `print()`, `printf()`, `log()` calls

## FiveM-Specific Optimizations

### Native Call Caching

FiveM natives have overhead crossing the Lua/C boundary. FLAO detects repeated calls and suggests caching:

**Before:**
```lua
Citizen.CreateThread(function()
    while true do
        local coords = GetEntityCoords(PlayerPedId())
        local heading = GetEntityHeading(PlayerPedId())
        local model = GetEntityModel(PlayerPedId())
        Wait(0)
    end
end)
```

**After (with --fix):**
```lua
Citizen.CreateThread(function()
    while true do
        local ped = PlayerPedId()
        local coords = GetEntityCoords(ped)
        local heading = GetEntityHeading(ped)
        local model = GetEntityModel(ped)
        Wait(0)
    end
end)
```

### Distance Calculation

`GetDistanceBetweenCoords` is expensive. FLAO suggests using vector math instead:

**Before:**
```lua
local dist = GetDistanceBetweenCoords(x1, y1, z1, x2, y2, z2, true)
```

**After (manual change suggested):**
```lua
local dist = #(vector3(x1, y1, z1) - vector3(x2, y2, z2))
-- Or with existing vectors:
local dist = #(coords1 - coords2)
```

This is ~40% faster and more readable.

### String Concat Fix (Experimental)

The `--experimental` flag enables automatic transformation of O(n²) string concatenation:

**Before:**
```lua
local result = ""
for i = 1, 10 do
    result = result .. GetLine(i)
end
```

**After:**
```lua
local _result_parts = {}
for i = 1, 10 do
    _result_parts[#_result_parts+1] = GetLine(i)
end
local result = table.concat(_result_parts)
```

## Resource Discovery

FLAO automatically discovers FiveM resources by looking for `fxmanifest.lua` or `__resource.lua` files:

```
resources/
├── [qb]/
│   ├── qb-core/
│   │   ├── fxmanifest.lua
│   │   ├── client/main.lua
│   │   └── server/main.lua
│   └── qb-inventory/
│       └── ...
├── standalone/
│   └── my-resource/
│       ├── fxmanifest.lua
│       └── client.lua
```

Use `--direct` to process individual files or folders without resource structure.

## Safety Measures

FLAO creates `.flao-bak` backup files before modifying any script. On first fix run, it also creates a full zip backup automatically.

```bash
# Make a full backup before modifications
python fivem_lua_lint.py /path/to/resources --backup-all-scripts

# Restore from backups
python fivem_lua_lint.py /path/to/resources --revert

# List all backups
python fivem_lua_lint.py /path/to/resources --list-backups
```

## Performance Tips for FiveM Scripts

1. **Cache `PlayerPedId()` once per tick** - It's called frequently, cache it at the start of your loop
2. **Use vector math for distances** - `#(v1 - v2)` instead of `GetDistanceBetweenCoords()`
3. **Avoid `Wait(0)` when possible** - Use longer intervals if you don't need every-frame updates
4. **Cache hash keys** - `GetHashKey()` does string hashing; cache results for repeated lookups
5. **Use `table.concat()` for string building** - Avoid `s = s .. x` in loops

## Example Output

```
$ python fivem_lua_lint.py ./resources --fix --report report.html

Scanning: ./resources
Found 45 resources with 312 Lua files

Analyzing with 8 workers...
[100.0%] 312/312 | ETA: 0s

============================================================
ANALYSIS SUMMARY
============================================================

  GREEN  (auto-fixable):    847
  YELLOW (review needed):    23
  RED    (info only):        12
  DEBUG  (logging):         156
  ----------------------------
  TOTAL:                   1038

Top issues by type:
  [G] repeated_PlayerPedId: 234
  [G] table_insert_append: 189
  [Y] distance_native: 23
  [G] repeated_GetEntityCoords: 156
```

## License

MIT License - See original ALAO project for attribution.

## Credits

- Original ALAO: Abraham (Priler)
- FiveM adaptation: ook3d
- Lua parser: [luaparser](https://pypi.org/project/luaparser/)
