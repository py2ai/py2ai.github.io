---
layout: post
title: "PS Smart Agent - Troubleshooting Shell Integration"
date: 2026-03-22
categories: [AI, VS Code, Tutorial]
featured-img: ps-smart-agent/troubleshooting
description: "Fix common issues with PS Smart Agent terminal and command execution, including shell integration problems."
keywords:
- PS Smart Agent
- troubleshooting
- shell integration
- terminal
- commands
- fix
---

# Troubleshooting Shell Integration

Having issues with command execution in PS Smart Agent? This guide covers common problems and solutions.

## Common Issues

### Commands Not Executing

**Symptoms:**
- Commands hang or timeout
- No output displayed
- "Command failed" errors

**Solutions:**

1. **Enable Shell Integration**
   - Open VS Code settings
   - Search for "terminal.integrated.shellIntegration.enabled"
   - Set to `true`

2. **Check Terminal Profile**
   ```json
   {
     "terminal.integrated.defaultProfile.windows": "PowerShell",
     "terminal.integrated.defaultProfile.linux": "bash",
     "terminal.integrated.defaultProfile.osx": "zsh"
   }
   ```

3. **Restart VS Code**
   - After changing settings, restart VS Code
   - Close all terminals and try again

### Permission Denied

**Symptoms:**
- "Permission denied" errors
- Commands fail silently

**Solutions:**

1. **Check file permissions (Linux/macOS)**
   ```bash
   chmod +x your-script.sh
   ```

2. **Run with appropriate permissions**
   - Use `sudo` when necessary
   - Or adjust user permissions

### Command Timeout

**Symptoms:**
- Commands take too long
- Process appears stuck

**Solutions:**

1. **Increase timeout**
   - Settings > Terminal > Command Timeout
   - Set to higher value (e.g., 300 seconds)

2. **Add to timeout allowlist**
   ```json
   {
     "ps.code.commandTimeoutAllowlist": [
       "npm install",
       "pip install",
       "cargo build"
     ]
   }
   ```

### Shell Not Found

**Symptoms:**
- "Shell not found" error
- Terminal doesn't open

**Solutions:**

1. **Verify shell path**
   ```json
   {
     "terminal.integrated.profiles.windows": {
       "PowerShell": {
         "source": "PowerShell",
         "path": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"
       }
     }
   }
   ```

2. **Install missing shell**
   - Windows: PowerShell is included
   - Linux: `sudo apt install bash`
   - macOS: bash/zsh are included

## VS Code Settings

### Recommended Settings

```json
{
  "terminal.integrated.shellIntegration.enabled": true,
  "terminal.integrated.shellIntegration.suggestEnabled": true,
  "terminal.integrated.enableMultiLinePasteWarning": "never",
  "ps.code.commandExecutionTimeout": 0
}
```

## Debug Mode

Enable debug mode for more information:

1. Settings > PS Smart Agent > Debug
2. Check "Enable Debug Mode"
3. View Output > PS Smart Agent

## Still Having Issues?

1. Check the Output panel: View → Output → PS Smart Agent
2. Check Developer Tools: Help → Toggle Developer Tools → Console
3. Visit [pyshine.com](https://pyshine.com) for more tutorials and support

---

*Learn more at [pyshine.com](https://pyshine.com)*
