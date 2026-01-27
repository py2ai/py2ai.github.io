---
layout: post
title: "PyShine clipboard App for two PCs"
description: "A free App that lets you connect two computers and copy paste text and transfer files over the network"
featured-img: clipboard-20260227/clipboard-20260227
keywords:
- clipboard
- Free transfer data
- wifi data transfer
- Mac and windows pc data transfer
- windows pc data transfer
- instant copy paste text 
---

# PyShine Clipboard

Seamlessly copy and paste **text, files, and folders** between devices on the same local network —  
**no cloud, no login, no tracking**.

<div class="video-container">
  <iframe 
    width="560" 
    height="315" 
    src="https://www.youtube.com/embed/Dw-Y_B5mzRQ" 
    title="YouTube video player" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen>
  </iframe>
</div>


Think *AirDrop*, but **cross‑platform**, **private**, and **under your control**.

---

## Features

- Real-time clipboard sync
- File & folder transfer with progress bar
- Works fully offline (Wifi or LAN)
- Cross-platform (Windows & macOS)
- Fast, lightweight, and secure

---

## Download Installers

### Windows (64-bit)

- **Installer (.exe)**  

[Download for Windows]({{ site.baseurl }}/assets/downloads/pyshne_clipboard.exe)

> Compatible with Windows 10 and above

---

### macOS
- **Intel Macs**  

[Download for macOS]({{ site.baseurl }}/assets/downloads/pyshine_clipboard.zip)

Unzip pyshine_clipboard on macOS and then press option key and then click and select Open.

> Compatible with macOS 11 (Big Sur) and above

⚠️ On first launch, you may need to:
- Right-click → Open
- Allow network access when prompted

Use ipconfig on windows pc to find your IP address and use ifconfig en0 on MacOs to find the IP. Open both apps on both PCs, and enter the target IP. For example in Windows PC app the target IP is of MacOs IP and vice-versa.

---

## How It Works (Quick)

1. Install Clipboard Pro on both machines
2. Make sure both are on the same Wi‑Fi / LAN
3. Launch the app on both devices
4. Copy on one → Paste on the other

---

## Complete Code

```python
#!/usr/bin/env python3
"""
PyShine Clipboard Pro - Complete Cross-Platform Solution
With Real-Time Progress Tracking for All Transfers
Fixed for macOS clipboard detection with circular progress indicators
DIRECT CONNECTION VERSION - Only communication between two PCs

"""

import sys
import os
import socket
import threading
import time
import hashlib
import uuid
import platform
import json
import struct
import base64
import mimetypes
import tempfile
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import queue
import pyperclip
import pyclip
import shutil
import netifaces
import ipaddress
import subprocess
import math
import traceback
from pathlib import Path
import platform


# ------------------ Configuration ------------------
TCP_PORT = 6000
BUFFER_SIZE = 8192
CHUNK_SIZE = 65536
DEVICE_ID = str(uuid.uuid4())[:8]
HOST_OS = platform.system()

# Clipboard check intervals - increased for macOS stability
TEXT_CHECK_INTERVAL = 1.5  # Increased from 1.0 for macOS

# Set proper clipboard backend for macOS
CLIPBOARD_BACKEND = None
if HOST_OS == "Darwin":
    # Try pyclip first (more reliable on macOS)
    try:
        import pyclip
        CLIPBOARD_BACKEND = "pyclip"
        print("Using pyclip for macOS clipboard")
    except ImportError:
        try:
            # Try to set pyperclip to use pbcopy/pbpaste
            pyperclip.set_clipboard("pbcopy")
            CLIPBOARD_BACKEND = "pyperclip"
            print("Using pyperclip with pbcopy for macOS clipboard")
        except:
            CLIPBOARD_BACKEND = "subprocess"
            print("Using subprocess for macOS clipboard")
elif HOST_OS == "Windows":
    CLIPBOARD_BACKEND = "pyperclip"
    try:
        import win32clipboard
    except ImportError:
        pass
else:  # Linux
    CLIPBOARD_BACKEND = "pyperclip"

def setup_application_icon(root):
    """
    Setup application icon with reliable fallbacks.
    Creates .pyshine_clipboard folder in user directory and stores icon there.
    """
    try:
        # Create application folder in user directory
        app_folder = Path.home() / ".pyshine_clipboard"
        app_folder.mkdir(exist_ok=True)
        
        # Define icon paths
        icon_paths = {
            'png': app_folder / "icon.png",
            'ico': app_folder / "icon.ico",
        }
        
        # Check if we're running as PyInstaller executable
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            # We're running as a bundled executable
            bundle_dir = Path(sys._MEIPASS)
            
            # Look for icon in bundle
            possible_bundle_icons = [
                bundle_dir / "icon_clean.png",
                bundle_dir / "icon.png",
                bundle_dir / "app.ico",
                bundle_dir / "icon.ico",
            ]
            
            bundle_icon_found = None
            for bundle_icon in possible_bundle_icons:
                if bundle_icon.exists():
                    bundle_icon_found = bundle_icon
                    break
            
            if bundle_icon_found:
                # Copy icon from bundle to user folder
                try:
                    shutil.copy2(bundle_icon_found, icon_paths['png'])
                    print(f"Icon copied from bundle to: {icon_paths['png']}")
                except Exception as e:
                    print(f"Failed to copy icon from bundle: {e}")
        
        # Now try to load icon from user folder
        icon_loaded = False
        
        # Try PNG first (works on all platforms)
        if icon_paths['png'].exists():
            try:
                icon = tk.PhotoImage(file=str(icon_paths['png']))
                root.iconphoto(True, icon)
                root._icon = icon  # Keep reference
                print(f"Icon loaded from: {icon_paths['png']}")
                icon_loaded = True
            except Exception as e:
                print(f"Failed to load PNG icon: {e}")
        
        # Try ICO for Windows
        if not icon_loaded and platform.system() == "Windows" and icon_paths['ico'].exists():
            try:
                root.iconbitmap(str(icon_paths['ico']))
                print(f"Windows ICO icon loaded from: {icon_paths['ico']}")
                icon_loaded = True
            except Exception as e:
                print(f"Failed to load ICO icon: {e}")
        
        # Try to convert PNG to ICO if needed (Windows only)
        if not icon_loaded and platform.system() == "Windows" and icon_paths['png'].exists():
            try:
                # Convert PNG to ICO
                from PIL import Image
                img = Image.open(icon_paths['png'])
                img.save(icon_paths['ico'], format='ICO')
                root.iconbitmap(str(icon_paths['ico']))
                print(f"Converted PNG to ICO and loaded: {icon_paths['ico']}")
                icon_loaded = True
            except ImportError:
                print("PIL not available for PNG to ICO conversion")
            except Exception as e:
                print(f"Failed to convert PNG to ICO: {e}")
        
        # Final fallback: try to load from any location
        if not icon_loaded:
            print("Trying fallback icon locations...")
            fallback_paths = [
                Path.cwd() / "icon_clean.png",
                Path.cwd() / "icon.png",
                Path(sys.executable).parent / "icon_clean.png",
                Path(sys.executable).parent / "icon.png",
            ]
            
            for fallback_path in fallback_paths:
                if fallback_path.exists():
                    try:
                        if fallback_path.suffix.lower() == '.png':
                            icon = tk.PhotoImage(file=str(fallback_path))
                            root.iconphoto(True, icon)
                            root._icon = icon
                            print(f"Fallback icon loaded from: {fallback_path}")
                            icon_loaded = True
                            # Copy to user folder for future use
                            shutil.copy2(fallback_path, icon_paths['png'])
                            break
                    except Exception as e:
                        print(f"Failed to load fallback icon {fallback_path}: {e}")
        
        if not icon_loaded:
            print("Could not load any icon, using Tkinter default")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error in setup_application_icon: {e}")
        return False
# ------------------ Enhanced Network Utilities ------------------
class NetworkUtils:
    """Network utilities for cross-platform compatibility"""
    
    @staticmethod
    def get_all_network_interfaces():
        """Get all network interfaces with IP addresses"""
        interfaces = []
        try:
            for iface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(iface)
                if netifaces.AF_INET in addrs:
                    for addr in addrs[netifaces.AF_INET]:
                        if 'addr' in addr and addr['addr'] != '127.0.0.1':
                            interfaces.append({
                                'name': iface,
                                'ip': addr['addr'],
                                'netmask': addr.get('netmask', '255.255.255.0'),
                                'broadcast': addr.get('broadcast', '')
                            })
        except Exception as e:
            print(f"Error getting network interfaces: {e}")
        return interfaces
    
    @staticmethod
    def get_local_ip():
        """Get local IP address"""
        try:
            # Get local IP from socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0)
            try:
                # Doesn't need to be reachable
                s.connect(('10.255.255.255', 1))
                local_ip = s.getsockname()[0]
            except Exception:
                local_ip = '127.0.0.1'
            finally:
                s.close()
            return local_ip
        except Exception:
            return '127.0.0.1'
    
    @staticmethod
    def validate_ip_address(ip):
        """Validate IP address format"""
        try:
            socket.inet_aton(ip)
            return True
        except socket.error:
            return False
    
    @staticmethod
    def is_valid_port(port):
        """Validate port number"""
        try:
            port = int(port)
            return 1 <= port <= 65535
        except ValueError:
            return False

# ------------------ Circular Progress Widget ------------------
class CircularProgress(tk.Canvas):
    """Circular spinning progress indicator"""
    
    def __init__(self, parent, size=40, thickness=4, color="#2196F3", bg_color="#f0f0f0"):
        self.size = size
        self.thickness = thickness
        self.color = color
        self.bg_color = bg_color
        
        tk.Canvas.__init__(self, parent, 
                          width=size, 
                          height=size, 
                          bg=bg_color,
                          highlightthickness=0)
        
        self.center = size // 2
        self.radius = (size - thickness) // 2
        
        # Draw background circle
        self.bg_circle = self.create_oval(
            self.center - self.radius,
            self.center - self.radius,
            self.center + self.radius,
            self.center + self.radius,
            outline=bg_color,
            fill=bg_color,
            width=0
        )
        
        # Create progress arc
        self.progress_arc = None
        self.angle = 0
        self.animation_id = None
        self.is_running = False
        
        # Start position markers
        self.start_markers = []
        for i in range(12):
            angle = i * 30
            rad = math.radians(angle)
            x1 = self.center + (self.radius - 2) * math.cos(rad)
            y1 = self.center + (self.radius - 2) * math.sin(rad)
            x2 = self.center + self.radius * math.cos(rad)
            y2 = self.center + self.radius * math.sin(rad)
            
            marker = self.create_line(x1, y1, x2, y2, 
                                     fill=self.color, 
                                     width=2,
                                     state='hidden')
            self.start_markers.append(marker)
    
    def start(self):
        """Start the spinning animation"""
        self.is_running = True
        self._animate()
    
    def stop(self):
        """Stop the spinning animation"""
        self.is_running = False
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None
        
        # Hide all markers
        for marker in self.start_markers:
            self.itemconfig(marker, state='hidden')
    
    def _animate(self):
        """Animate the circular progress"""
        if not self.is_running:
            return
        
        # Clear previous arc
        if self.progress_arc:
            self.delete(self.progress_arc)
        
        # Calculate arc coordinates
        start_angle = self.angle
        extent = 80  # Length of the arc
        
        # Draw new arc
        self.progress_arc = self.create_arc(
            self.center - self.radius,
            self.center - self.radius,
            self.center + self.radius,
            self.center + self.radius,
            start=start_angle,
            extent=extent,
            outline=self.color,
            width=self.thickness,
            style=tk.ARC
        )
        
        # Update angle for next frame
        self.angle = (self.angle + 10) % 360
        
        # Animate markers
        for i, marker in enumerate(self.start_markers):
            alpha = (math.sin(math.radians(self.angle + i * 30)) + 1) / 2
            r = int(255 * alpha)
            g = int(255 * alpha)
            b = int(255 * alpha)
            color = f'#{r:02x}{g:02x}{b:02x}'
            self.itemconfig(marker, fill=color, state='normal' if alpha > 0.3 else 'hidden')
        
        # Schedule next animation frame
        self.animation_id = self.after(50, self._animate)
    
    def set_color(self, color):
        """Change the progress color"""
        self.color = color
        if self.progress_arc:
            self.itemconfig(self.progress_arc, outline=color)
        for marker in self.start_markers:
            self.itemconfig(marker, fill=color)

# ------------------ Enhanced Clipboard Manager for macOS ------------------
class ClipboardManager:
    """Enhanced clipboard manager with macOS-specific fixes"""
    
    def __init__(self, log_callback):
        self.log_callback = log_callback
        self.last_hash = None
        self.last_text = ""
        self.clipboard_lock = threading.Lock()
        
        # Initialize clipboard for macOS
        if HOST_OS == "Darwin":
            self._init_macos_clipboard()
    
    def _init_macos_clipboard(self):
        """Initialize macOS clipboard"""
        try:
            # Test clipboard access
            test_text = "test"
            success = self._set_clipboard_text_macos(test_text)
            if success:
                retrieved = self._get_clipboard_text_macos()
                if retrieved == test_text:
                    self.log_callback("[CLIPBOARD] macOS clipboard initialized successfully")
                else:
                    self.log_callback("[CLIPBOARD] Warning: macOS clipboard verification failed")
            else:
                self.log_callback("[CLIPBOARD] Warning: Could not initialize macOS clipboard")
        except Exception as e:
            self.log_callback(f"[CLIPBOARD] Error initializing macOS clipboard: {e}")
    
    def _get_clipboard_text_macos(self):
        """Get clipboard text specifically for macOS with multiple fallbacks"""
        with self.clipboard_lock:
            text = ""
            try:
                # Try pyclip first (most reliable on macOS)
                if CLIPBOARD_BACKEND == "pyclip":
                    try:
                        text = pyclip.paste(text=True)
                        if text:
                            return text
                    except UnicodeDecodeError:
                        # Silently skip if pyclip can't decode. Other methods might succeed.
                        pass  # ⬅️ Changed from logging and returning
                    except Exception as e:
                        self.log_callback(f"[CLIPBOARD] pyclip failed: {e}")
                
                # Try pyperclip
                try:
                    text = pyperclip.paste()
                    if text and text.strip():
                        return text
                except UnicodeDecodeError:
                    return ""
                except Exception as e:
                    self.log_callback(f"[CLIPBOARD] pyperclip failed: {e}")
                
                # Try pbpaste command directly
                try:
                    result = subprocess.run(
                        ['pbpaste'],
                        capture_output=True,
                        timeout=2,
                        env={'LANG': 'en_US.UTF-8'}
                    )
                    if result.returncode == 0 and result.stdout:
                        # Try to decode, but ignore if it fails
                        try:
                            text = result.stdout.decode('utf-8', errors='ignore')
                            if text and text.strip():
                                return text
                        except UnicodeDecodeError:
                            return ""
                except Exception as e:
                    self.log_callback(f"[CLIPBOARD] pbpaste failed: {e}")
                
                # Last resort: use AppleScript
                try:
                    script = '''
                    try
                        get the clipboard as text
                    on error
                        ""
                    end try
                    '''
                    result = subprocess.run(
                        ['osascript', '-e', script],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        text = result.stdout.strip()
                        return text
                except Exception as e:
                    self.log_callback(f"[CLIPBOARD] AppleScript failed: {e}")
                
            except Exception as e:
                self.log_callback(f"[CLIPBOARD] All macOS clipboard methods failed: {e}")
            
            return ""
    def __get_clipboard_text_macos(self):
        """Get clipboard text specifically for macOS with multiple fallbacks"""
        with self.clipboard_lock:
            text = ""
            try:
                # Try pyclip first (most reliable on macOS)
                if CLIPBOARD_BACKEND == "pyclip":
                    try:
                        text = pyclip.paste(text=True)
                        if text:
                            return text
                    except Exception as e:
                        self.log_callback(f"[CLIPBOARD] pyclip failed: {e}")
                
                # Try pyperclip
                try:
                    text = pyperclip.paste()
                    if text and text.strip():
                        return text
                except Exception as e:
                    self.log_callback(f"[CLIPBOARD] pyperclip failed: {e}")
                
                # Try pbpaste command directly
                try:
                    result = subprocess.run(
                        ['pbpaste'],
                        capture_output=True,
                        text=True,
                        timeout=2,
                        env={'LANG': 'en_US.UTF-8'}  # Set locale for consistency
                    )
                    if result.returncode == 0 and result.stdout:
                        text = result.stdout
                        return text
                except Exception as e:
                    self.log_callback(f"[CLIPBOARD] pbpaste failed: {e}")
                
                # Last resort: use AppleScript
                try:
                    script = '''
                    try
                        get the clipboard as text
                    on error
                        ""
                    end try
                    '''
                    result = subprocess.run(
                        ['osascript', '-e', script],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        text = result.stdout.strip()
                        return text
                except Exception as e:
                    self.log_callback(f"[CLIPBOARD] AppleScript failed: {e}")
                
            except Exception as e:
                self.log_callback(f"[CLIPBOARD] All macOS clipboard methods failed: {e}")
            
            return ""
    
    def _set_clipboard_text_macos(self, text):
        """Set clipboard text specifically for macOS with multiple fallbacks"""
        with self.clipboard_lock:
            try:
                # Try pyclip first
                if CLIPBOARD_BACKEND == "pyclip":
                    try:
                        pyclip.copy(text)
                        # Verify
                        time.sleep(0.1)
                        verify = pyclip.paste(text=True)
                        if verify == text:
                            return True
                    except Exception as e:
                        self.log_callback(f"[CLIPBOARD] pyclip copy failed: {e}")
                
                # Try pyperclip
                try:
                    pyperclip.copy(text)
                    # Verify
                    time.sleep(0.1)
                    verify = pyperclip.paste()
                    if verify == text:
                        return True
                except Exception as e:
                    self.log_callback(f"[CLIPBOARD] pyperclip copy failed: {e}")
                
                # Try pbcopy command directly
                try:
                    result = subprocess.run(
                        ['pbcopy'],
                        input=text.encode('utf-8'),
                        timeout=2,
                        env={'LANG': 'en_US.UTF-8'}
                    )
                    if result.returncode == 0:
                        # Verify with pbpaste
                        time.sleep(0.1)
                        verify_result = subprocess.run(
                            ['pbpaste'],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        if verify_result.returncode == 0 and verify_result.stdout == text:
                            return True
                except Exception as e:
                    self.log_callback(f"[CLIPBOARD] pbcopy failed: {e}")
                
                # Last resort: use AppleScript
                try:
                    # Escape quotes in text
                    escaped_text = text.replace('"', '\\"')
                    script = f'set the clipboard to "{escaped_text}"'
                    result = subprocess.run(
                        ['osascript', '-e', script],
                        timeout=2
                    )
                    if result.returncode == 0:
                        return True
                except Exception as e:
                    self.log_callback(f"[CLIPBOARD] AppleScript copy failed: {e}")
                
            except Exception as e:
                self.log_callback(f"[CLIPBOARD] All macOS clipboard set methods failed: {e}")
            
            return False
    
    def get_clipboard_text(self):
        """Safely get clipboard text for all platforms"""
        if HOST_OS == "Darwin":
            return self._get_clipboard_text_macos()
        else:
            try:
                with self.clipboard_lock:
                    return pyperclip.paste()
            except Exception as e:
                self.log_callback(f"[CLIPBOARD] Error getting text: {e}")
                return ""
    
    def set_clipboard_text(self, text):
        """Safely set clipboard text for all platforms"""
        if HOST_OS == "Darwin":
            return self._set_clipboard_text_macos(text)
        else:
            try:
                with self.clipboard_lock:
                    pyperclip.copy(text)
                    # Verify
                    time.sleep(0.1)
                    verify = pyperclip.paste()
                    return verify == text
            except Exception as e:
                self.log_callback(f"[CLIPBOARD] Error setting text: {e}")
                return False
    
    def check_for_changes(self):
        """Check if clipboard text has changed with improved macOS handling"""
        try:
            current_text = self.get_clipboard_text()
            
            if not current_text or current_text.strip() == "":
                return None
            
            # Normalize text - remove extra whitespace that macOS might add
            current_text = current_text.strip()
            current_hash = hashlib.md5(current_text.encode()).hexdigest()
            
            if current_hash != self.last_hash:
                self.last_hash = current_hash
                self.last_text = current_text
                return current_text
            
            return None
        except Exception as e:
            self.log_callback(f"[CLIPBOARD] Error checking for changes: {e}")
            return None

# ------------------ Progress Manager ------------------
class ProgressManager:
    """Manages progress tracking for file transfers"""
    
    def __init__(self, gui_callback):
        self.gui_callback = gui_callback
        self.active_transfers = {}
        self.lock = threading.Lock()
        
    def start_transfer(self, transfer_id, filename, total_size, transfer_type="send"):
        """Start tracking a new transfer"""
        with self.lock:
            self.active_transfers[transfer_id] = {
                'filename': filename,
                'total_size': total_size,
                'current_size': 0,
                'start_time': time.time(),
                'transfer_type': transfer_type,
                'last_update': time.time(),
                'completed': False
            }
            
            # Notify GUI
            self.gui_callback('transfer_started', {
                'transfer_id': transfer_id,
                'filename': filename,
                'total_size': total_size,
                'transfer_type': transfer_type
            })
    
    def update_progress(self, transfer_id, bytes_transferred):
        """Update progress for a transfer"""
        with self.lock:
            if transfer_id in self.active_transfers:
                transfer = self.active_transfers[transfer_id]
                transfer['current_size'] += bytes_transferred
                transfer['last_update'] = time.time()
                
                # Calculate progress percentage (for internal use only)
                if transfer['total_size'] > 0:
                    percentage = (transfer['current_size'] / transfer['total_size']) * 100
                else:
                    percentage = 0
                
                # Notify GUI - simplified for circular progress
                self.gui_callback('progress_updated', {
                    'transfer_id': transfer_id,
                    'percentage': percentage,
                    'current_size': transfer['current_size'],
                    'total_size': transfer['total_size'],
                    'transfer_type': transfer['transfer_type']
                })
    
    def complete_transfer(self, transfer_id, success=True, error_msg=None):
        """Mark a transfer as completed"""
        with self.lock:
            if transfer_id in self.active_transfers:
                transfer = self.active_transfers[transfer_id]
                transfer['completed'] = True
                transfer['end_time'] = time.time()
                transfer['success'] = success
                transfer['error'] = error_msg
                
                # Calculate total time
                total_time = transfer['end_time'] - transfer['start_time']
                
                # Notify GUI
                self.gui_callback('transfer_completed', {
                    'transfer_id': transfer_id,
                    'filename': transfer['filename'],
                    'success': success,
                    'error_msg': error_msg,
                    'total_time': total_time,
                    'total_size': transfer['total_size']
                })
                
                # Keep for a while then remove
                threading.Timer(10.0, self._remove_transfer, args=[transfer_id]).start()
    
    def _remove_transfer(self, transfer_id):
        """Remove a completed transfer"""
        with self.lock:
            if transfer_id in self.active_transfers:
                del self.active_transfers[transfer_id]
    
    def format_size(self, size_bytes):
        """Format file size in human-readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes/(1024*1024):.1f} MB"
        else:
            return f"{size_bytes/(1024*1024*1024):.2f} GB"
    
    def get_active_transfers(self):
        """Get list of active transfers"""
        with self.lock:
            return list(self.active_transfers.items())

# ------------------ File Transfer Manager ------------------
class FileTransferManager:
    """Handles all file and folder transfers"""
    
    @staticmethod
    def zip_folder(folder_path, progress_callback=None):
        """Zip a folder for transfer with progress tracking"""
        try:
            folder_path = Path(folder_path)
            if not folder_path.exists() or not folder_path.is_dir():
                return None, "Not a valid folder"
            
            # Count total files for progress
            file_list = []
            total_size = 0
            for file_path in folder_path.rglob('*'):
                if file_path.is_file():
                    file_list.append(file_path)
                    total_size += file_path.stat().st_size
            
            if progress_callback:
                progress_callback(0, f"Preparing {len(file_list)} files...")
            
            # Create temp zip file
            temp_dir = Path(tempfile.gettempdir())
            zip_path = temp_dir / f"{folder_path.name}_{int(time.time())}.zip"
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for i, file_path in enumerate(file_list):
                    arcname = file_path.relative_to(folder_path)
                    zipf.write(file_path, arcname)
                    
                    if progress_callback and (i % 10 == 0 or i == len(file_list) - 1):
                        progress = (i + 1) / len(file_list) * 100
                        progress_callback(progress, f"Compressing: {file_path.name}")
            
            if progress_callback:
                progress_callback(100, "Compression complete")
            
            return zip_path, None
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def get_file_info(filepath):
        """Get file information for transfer"""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return None, "File does not exist"
            
            info = {
                'name': filepath.name,
                'size': filepath.stat().st_size,
                'modified': filepath.stat().st_mtime,
                'is_dir': filepath.is_dir(),
                'is_file': filepath.is_file(),
                'path': str(filepath)
            }
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(filepath))
            if not mime_type:
                if filepath.is_dir():
                    mime_type = 'application/x-directory'
                else:
                    mime_type = 'application/octet-stream'
            
            info['mime_type'] = mime_type
            
            # Get folder info if it's a directory
            if filepath.is_dir():
                file_count = 0
                total_size = 0
                for item in filepath.rglob('*'):
                    if item.is_file():
                        file_count += 1
                        total_size += item.stat().st_size
                
                info['file_count'] = file_count
                info['total_size'] = total_size
            
            return info, None
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def format_size(size_bytes):
        """Format file size in human-readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes/(1024*1024):.1f} MB"
        else:
            return f"{size_bytes/(1024*1024*1024):.2f} GB"
    
    @staticmethod
    def open_file(filepath):
        """Open file with default application"""
        try:
            filepath = Path(filepath)
            if HOST_OS == 'Windows':
                os.startfile(filepath)
            elif HOST_OS == 'Darwin':
                subprocess.run(['open', filepath], check=False)
            else:
                subprocess.run(['xdg-open', filepath], check=False)
        except Exception as e:
            return str(e)
        return None
    
    @staticmethod
    def open_folder(folderpath):
        """Open folder in file browser"""
        try:
            folderpath = Path(folderpath)
            if HOST_OS == 'Windows':
                os.startfile(folderpath)
            elif HOST_OS == 'Darwin':
                subprocess.run(['open', folderpath], check=False)
            else:
                subprocess.run(['xdg-open', folderpath], check=False)
        except Exception as e:
            return str(e)
        return None

# ------------------ Clipboard Monitor ------------------
class ClipboardMonitor:
    """Monitor clipboard for changes with improved macOS handling"""
    
    def __init__(self, clipboard_callback, log_callback):
        self.clipboard_callback = clipboard_callback
        self.log_callback = log_callback
        self.running = False
        self.clipboard_manager = ClipboardManager(log_callback)
        self.last_sent_hash = None
        self.monitor_lock = threading.Lock()
        
    def start(self):
        """Start monitoring"""
        self.running = True
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        self.log_callback("[MONITOR] Clipboard monitoring started")
        
    def _monitor_loop(self):
        """Main monitoring loop with improved error handling"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running:
            try:
                with self.monitor_lock:
                    new_text = self.clipboard_manager.check_for_changes()
                    
                    if new_text and self.last_sent_hash != self.clipboard_manager.last_hash:
                        self.last_sent_hash = self.clipboard_manager.last_hash
                        
                        clipboard_item = {
                            'type': 'text',
                            'data': new_text,
                            'mime_type': 'text/plain',
                            'timestamp': time.time(),
                            'hash': self.clipboard_manager.last_hash
                        }
                        
                        self.clipboard_callback(clipboard_item)
                        
                        # Reset error counter on success
                        consecutive_errors = 0
                
                time.sleep(TEXT_CHECK_INTERVAL)
                
            except Exception as e:
                consecutive_errors += 1
                self.log_callback(f"[MONITOR] Error in loop: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    self.log_callback("[MONITOR] Too many errors, restarting clipboard manager...")
                    # Reinitialize clipboard manager
                    self.clipboard_manager = ClipboardManager(self.log_callback)
                    consecutive_errors = 0
                
                # Exponential backoff on errors
                sleep_time = TEXT_CHECK_INTERVAL * (2 ** min(consecutive_errors, 3))
                time.sleep(sleep_time)
                
    def stop(self):
        """Stop monitoring"""
        self.running = False
        self.log_callback("[MONITOR] Clipboard monitoring stopped")

# ------------------ TCP Server (Receiver with Progress) ------------------
class ClipboardReceiver:
    """Receive clipboard data with progress tracking"""
    
    def __init__(self, port, data_callback, log_callback, progress_callback):
        self.port = port
        self.data_callback = data_callback
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.running = False
        self.server_socket = None
        
    def start(self):
        """Start TCP server"""
        self.running = True
        server_thread = threading.Thread(target=self._server_loop, daemon=True)
        server_thread.start()
        self.log_callback(f"[RECEIVER] Listening on TCP port {self.port}")
        
    def _server_loop(self):
        """Main server loop"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(10)
            self.server_socket.settimeout(1)
            
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        self.log_callback(f"[RECEIVER] Accept error: {e}")
                        
        except Exception as e:
            self.log_callback(f"[RECEIVER] Server error: {e}")
            
    def _handle_client(self, client_socket, client_address):
        """Handle incoming connection with progress tracking"""
        ip = client_address[0]
        transfer_id = f"recv_{ip}_{int(time.time())}"
        
        try:
            client_socket.settimeout(30)
            
            # Receive header
            header_data = self._receive_with_progress(client_socket, 4, transfer_id, 0, 0)
            if not header_data or len(header_data) < 4:
                return
                
            header_len = struct.unpack('!I', header_data)[0]
            
            # Receive header JSON
            header_json = self._receive_with_progress(client_socket, header_len, transfer_id, 0, 0)
            if not header_json:
                return
                
            header = json.loads(header_json.decode('utf-8'))
            data_type = header.get('type', 'text')
            total_size = header.get('size', 0)
            transfer_type = header.get('transfer_type', 'single')
            filename = header.get('filename', f'file_{int(time.time())}')
            is_folder = header.get('is_folder', False)
            
            # Start progress tracking for large files
            if total_size > 1024 * 1024:  # > 1MB
                self.progress_callback('start_transfer', {
                    'transfer_id': transfer_id,
                    'filename': filename,
                    'total_size': total_size,
                    'transfer_type': 'receive',
                    'source_ip': ip
                })
                
                self.log_callback(f"[RECEIVER] Receiving {filename} ({FileTransferManager.format_size(total_size)}) from {ip}")
            
            # Receive data with progress tracking
            data = self._receive_with_progress(client_socket, total_size, transfer_id, total_size, 100)
            
            if data:
                if data_type == 'text':
                    text = data.decode('utf-8', errors='ignore')
                    self.data_callback({
                        'type': 'text',
                        'data': text,
                        'source_ip': ip,
                        'timestamp': time.time()
                    })
                    
                elif data_type == 'file':
                    mime_type = header.get('mime_type', 'application/octet-stream')
                    
                    file_data = {
                        'type': 'file',
                        'data': data,
                        'filename': filename,
                        'mime_type': mime_type,
                        'is_folder': is_folder,
                        'transfer_type': transfer_type,
                        'source_ip': ip,
                        'timestamp': time.time()
                    }
                    
                    # Add original folder name if it's a folder
                    if is_folder:
                        file_data['original_folder_name'] = header.get('original_folder_name', filename)
                    
                    self.data_callback(file_data)
            
            # Complete progress tracking
            if total_size > 1024 * 1024:
                self.progress_callback('complete_transfer', {
                    'transfer_id': transfer_id,
                    'success': True,
                    'error_msg': None
                })
                    
        except Exception as e:
            self.log_callback(f"[RECEIVER] Client error from {ip}: {e}")
            if total_size > 1024 * 1024:
                self.progress_callback('complete_transfer', {
                    'transfer_id': transfer_id,
                    'success': False,
                    'error_msg': str(e)
                })
        finally:
            try:
                client_socket.close()
            except:
                pass
                
    def _receive_with_progress(self, sock, size, transfer_id, total_size, progress_start):
        """Receive data with progress tracking"""
        data = b''
        bytes_received = 0
        last_progress_update = 0
        
        while len(data) < size:
            try:
                chunk_size = min(CHUNK_SIZE, size - len(data))
                chunk = sock.recv(chunk_size)
                if not chunk:
                    break
                data += chunk
                bytes_received += len(chunk)
                
                # Update progress periodically
                if total_size > 0 and time.time() - last_progress_update > 0.1:
                    progress = (bytes_received / size) * 100
                    if progress - last_progress_update >= 1 or bytes_received % (64 * 1024) == 0:
                        self.progress_callback('update_progress', {
                            'transfer_id': transfer_id,
                            'bytes_transferred': len(chunk)
                        })
                        last_progress_update = progress
                
            except socket.timeout:
                break
            except:
                break
        return data
        
    def stop(self):
        """Stop server"""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        self.log_callback("[RECEIVER] Server stopped")

# ------------------ TCP Client (Sender with Progress) ------------------
class ClipboardSender:
    """Send clipboard data and files with progress tracking"""
    
    def __init__(self, log_callback, progress_callback):
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.transfer_manager = FileTransferManager()
        
    def send_text(self, ip, port, text):
        """Send text to device"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((ip, port))
            
            try:
                # Prepare header
                data = text.encode('utf-8')
                header = {
                    'type': 'text',
                    'size': len(data),
                    'timestamp': time.time()
                }
                
                header_json = json.dumps(header).encode('utf-8')
                header_len = struct.pack('!I', len(header_json))
                
                # Send header length
                sock.sendall(header_len)
                # Send header
                sock.sendall(header_json)
                # Send data
                sock.sendall(data)
                
                return True
                
            finally:
                sock.close()
                
        except Exception as e:
            self.log_callback(f"[SENDER] Error sending to {ip}:{port}: {e}")
            return False
            
    def send_file(self, ip, port, filepath, is_folder=False, transfer_id=None):
        """Send any file to device with progress tracking"""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return False, "File does not exist"
            
            # Get file info
            file_info, error = self.transfer_manager.get_file_info(filepath)
            if error:
                return False, error
            
            # Create transfer ID if not provided
            if not transfer_id:
                transfer_id = f"send_{ip}_{int(time.time())}"
            
            # Start progress tracking
            self.progress_callback('start_transfer', {
                'transfer_id': transfer_id,
                'filename': filepath.name,
                'total_size': file_info['size'],
                'transfer_type': 'send',
                'destination_ip': ip
            })
            
            self.log_callback(f"[SENDER] Starting transfer: {filepath.name} ({self.transfer_manager.format_size(file_info['size'])}) to {ip}")
            
            # Read file in chunks for progress tracking
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30)
            sock.connect((ip, port))
            
            try:
                # Prepare header
                header = {
                    'type': 'file',
                    'size': file_info['size'],
                    'filename': filepath.name,
                    'mime_type': file_info['mime_type'],
                    'is_folder': is_folder,
                    'transfer_type': 'single',
                    'timestamp': time.time()
                }
                
                # Add original folder name for folders
                if is_folder:
                    header['original_folder_name'] = filepath.name
                
                header_json = json.dumps(header).encode('utf-8')
                header_len = struct.pack('!I', len(header_json))
                
                # Send header length
                sock.sendall(header_len)
                # Send header
                sock.sendall(header_json)
                
                # Send file data in chunks with progress tracking
                bytes_sent = 0
                last_progress_update = 0
                
                with open(filepath, 'rb') as f:
                    while True:
                        chunk = f.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        
                        sock.sendall(chunk)
                        bytes_sent += len(chunk)
                        
                        # Update progress periodically
                        if time.time() - last_progress_update > 0.1:
                            progress = (bytes_sent / file_info['size']) * 100
                            if progress - last_progress_update >= 1:
                                self.progress_callback('update_progress', {
                                    'transfer_id': transfer_id,
                                    'bytes_transferred': len(chunk)
                                })
                                last_progress_update = progress
                
                # Complete progress tracking
                self.progress_callback('complete_transfer', {
                    'transfer_id': transfer_id,
                    'success': True,
                    'error_msg': None
                })
                
                self.log_callback(f"[SENDER] Transfer completed: {filepath.name}")
                return True, None
                
            finally:
                sock.close()
                
        except Exception as e:
            self.log_callback(f"[SENDER] Error sending file to {ip}:{port}: {e}")
            
            # Mark transfer as failed
            if transfer_id:
                self.progress_callback('complete_transfer', {
                    'transfer_id': transfer_id,
                    'success': False,
                    'error_msg': str(e)
                })
            
            return False, str(e)
    
    def send_folder_as_zip(self, ip, port, folder_path, transfer_id=None):
        """Send folder as ZIP archive with progress tracking"""
        try:
            folder_path = Path(folder_path)
            if not folder_path.exists() or not folder_path.is_dir():
                return False, "Not a valid folder"
            
            # Create transfer ID if not provided
            if not transfer_id:
                transfer_id = f"send_folder_{ip}_{int(time.time())}"
            
            # Get folder info
            folder_info, error = self.transfer_manager.get_file_info(folder_path)
            if error:
                return False, error
            
            # Start progress tracking for preparation
            self.progress_callback('start_transfer', {
                'transfer_id': transfer_id,
                'filename': folder_path.name,
                'total_size': folder_info.get('total_size', 0),
                'transfer_type': 'send',
                'destination_ip': ip,
                'is_folder': True
            })
            
            self.log_callback(f"[SENDER] Preparing folder: {folder_path.name} ({folder_info.get('file_count', 0)} files, {self.transfer_manager.format_size(folder_info.get('total_size', 0))})")
            
            # Create ZIP with progress callback
            def zip_progress_callback(progress, message):
                self.progress_callback('update_progress', {
                    'transfer_id': transfer_id,
                    'bytes_transferred': 0,
                    'message': message,
                    'preparation_progress': progress
                })
            
            zip_path, error = self.transfer_manager.zip_folder(folder_path, zip_progress_callback)
            if error:
                self.progress_callback('complete_transfer', {
                    'transfer_id': transfer_id,
                    'success': False,
                    'error_msg': f"Failed to create ZIP: {error}"
                })
                return False, f"Failed to create ZIP: {error}"
            
            try:
                # Send ZIP file
                success, error = self.send_file(ip, port, zip_path, is_folder=True, transfer_id=transfer_id)
                if success:
                    self.log_callback(f"[SENDER] Folder sent successfully: {folder_path.name}")
                else:
                    self.progress_callback('complete_transfer', {
                        'transfer_id': transfer_id,
                        'success': False,
                        'error_msg': error
                    })
                
                return success, error
            finally:
                # Clean up temp ZIP
                try:
                    os.remove(zip_path)
                except:
                    pass
                
        except Exception as e:
            self.log_callback(f"[SENDER] Error sending folder to {ip}:{port}: {e}")
            return False, str(e)

# ------------------ Circular Progress Dialog ------------------
class CircularProgressDialog:
    """Modal dialog showing circular progress indicator"""
    
    def __init__(self, parent, filename, total_size, transfer_type, destination):
        self.parent = parent
        self.filename = filename
        self.total_size = total_size
        self.transfer_type = transfer_type
        self.destination = destination
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"{'Sending' if transfer_type == 'send' else 'Receiving'} File")
        self.dialog.geometry("400x250")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - 200
        y = parent.winfo_y() + (parent.winfo_height() // 2) - 125
        self.dialog.geometry(f"+{x}+{y}")
        
        # Create widgets
        self._create_widgets()
        
        # Start time
        self.start_time = time.time()
        
    def _create_widgets(self):
        """Create dialog widgets"""
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_text = f"{'Sending' if self.transfer_type == 'send' else 'Receiving'}: {self.filename}"
        ttk.Label(main_frame, text=title_text, font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        # Destination
        dest_text = f"{'To' if self.transfer_type == 'send' else 'From'}: {self.destination}"
        ttk.Label(main_frame, text=dest_text).pack(anchor=tk.W, pady=(0, 10))
        
        # Size info
        size_str = FileTransferManager.format_size(self.total_size)
        ttk.Label(main_frame, text=f"Size: {size_str}").pack(anchor=tk.W, pady=(0, 10))
        
        # Progress frame
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(10, 10))
        
        # Circular progress indicator
        self.progress_circle = CircularProgress(progress_frame, size=60, thickness=5)
        self.progress_circle.pack(pady=10)
        self.progress_circle.start()
        
        # Status label
        self.status_label = ttk.Label(progress_frame, text="Transfer in progress...")
        self.status_label.pack(pady=(5, 0))
        
        # Cancel button
        self.cancel_button = ttk.Button(main_frame, text="Cancel", command=self.cancel)
        self.cancel_button.pack(pady=(10, 0))
        
    def update_progress(self, percentage, current_size, total_size):
        """Update progress display - simplified for circular progress"""
        # Update status text occasionally
        if int(percentage) % 10 == 0:
            current_str = FileTransferManager.format_size(current_size)
            total_str = FileTransferManager.format_size(total_size)
            self.status_label.config(text=f"{current_str} of {total_str}")
            
        # Update dialog
        self.dialog.update()
        
    def complete(self, success=True, error_msg=None):
        """Mark transfer as complete"""
        # Stop the circular animation
        self.progress_circle.stop()
        
        if success:
            # Change to success color (green)
            self.progress_circle.set_color("#4CAF50")
            
            # Draw a checkmark
            self.status_label.config(text="Transfer complete!")
            
            # Change cancel button to close
            self.cancel_button.config(text="Close", command=self.dialog.destroy)
            
            # Auto-close after 2 seconds
            self.dialog.after(2000, self.dialog.destroy)
        else:
            # Change to error color (red)
            self.progress_circle.set_color("#F44336")
            
            # Show error message
            self.status_label.config(text=f"Error: {error_msg[:50]}...")
            self.cancel_button.config(text="Close", command=self.dialog.destroy)
        
    def cancel(self):
        """Cancel the transfer"""
        self.dialog.destroy()
        
    def is_cancelled(self):
        """Check if dialog was closed"""
        try:
            return not self.dialog.winfo_exists()
        except:
            return True

# ------------------ Main Application GUI ------------------
class PyShineClipboardApp:
    """Main application with progress tracking - Direct Connection Version"""
    
    def __init__(self, root):
        self.root = root
        self.root.title(f"PyShine Clipboard Pro - v1.0")
        self.root.geometry("1000x850")
        
        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        # Initialize primary_received_dir
        self.primary_received_dir = None
        
        # Variables
        self.auto_sync_var = tk.BooleanVar(value=True)
        self.notify_var = tk.BooleanVar(value=True)
        self.target_ip_var = tk.StringVar(value="192.168.1.100")  # Default target IP
        self.target_port_var = tk.StringVar(value="6000")  # Default port
        self.local_ip_var = tk.StringVar(value="")
        
        # Data storage
        self.received_files = []
        
        # Progress tracking
        self.active_transfers = {}
        self.progress_dialogs = {}
        self.progress_manager = ProgressManager(self._progress_callback)
        
        # Clipboard state
        self.last_received_hash = None
        self.is_processing = False
        
        # Create queues
        self.log_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        
        # Load cached IP if available
        cached_ip, cached_port = self._load_cached_ip()
        if cached_ip:
            self.target_ip_var.set(cached_ip)
        if cached_port:
            self.target_port_var.set(cached_port)
        
        # Initialize components
        self._init_components()
        
        # Start services
        self._start_services()
        
        # Start GUI update loop
        self._update_gui_loop()
        
        # Setup window close handler - CRITICAL FIX
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self._log("=" * 60)
        self._log("PyShine Clipboard Pro - v1.0")
        self._log(f"Device ID: {DEVICE_ID}")
        self._log(f"Platform: {HOST_OS}")
        self._log(f"Clipboard Backend: {CLIPBOARD_BACKEND}")
        self._log("Mode: DIRECT CONNECTION BETWEEN TWO PCs")
        self._log("Enter target IP and port to connect")
        self._log("=" * 60)
        
        # Get and display local IP
        self._update_local_ip()
    def _get_cache_path(self):
        """Get path to cache file"""
        cache_dir = Path.home() / ".pyshine_clipboard"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / "last_connection.cache"

    def _load_cached_ip(self):
        """Load cached IP and port from file"""
        try:
            cache_file = self._get_cache_path()
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = f.read().strip()
                    if data:
                        # Format: "IP:PORT"
                        parts = data.split(':')
                        if len(parts) >= 2:
                            ip = parts[0]
                            port = parts[1]
                            return ip, port
                        elif len(parts) == 1:
                            return parts[0], "6000"  # Default port
        except Exception as e:
            self._log(f"Failed to load cached IP: {e}")
        return None, None

    def _save_cached_ip(self, ip=None, port=None):
        """Save current IP and port to cache file"""
        try:
            if ip is None:
                ip = self.target_ip_var.get().strip()
            if port is None:
                port = self.target_port_var.get().strip()
            
            # Validate
            if not ip or not port:
                return
            
            if not NetworkUtils.validate_ip_address(ip):
                return
            
            if not NetworkUtils.is_valid_port(port):
                return
            
            # Save to cache
            cache_file = self._get_cache_path()
            with open(cache_file, 'w') as f:
                f.write(f"{ip}:{port}")
                
            self._log(f"Cached connection: {ip}:{port}")
        except Exception as e:
            self._log(f"Failed to cache IP: {e}")
    def _update_local_ip(self):
        """Update local IP display"""
        local_ip = NetworkUtils.get_local_ip()
        self.local_ip_var.set(f"Enter the IP address of Target")
        # self._log(f"Local IP: {local_ip}")

    def _init_components(self):
        """Initialize GUI components"""
        # Main container
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Connection and controls
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=1)
        
        # Connection frame
        connection_frame = ttk.LabelFrame(left_panel, text="Direct Connection", padding=10)
        connection_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Local IP display
        ttk.Label(connection_frame, textvariable=self.local_ip_var, font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        # Target IP entry
        ip_frame = ttk.Frame(connection_frame)
        ip_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(ip_frame, text="Target IP:").pack(side=tk.LEFT, padx=(0, 5))
        self.target_ip_entry = ttk.Entry(ip_frame, textvariable=self.target_ip_var, width=15)
        self.target_ip_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        # Port entry
        port_frame = ttk.Frame(connection_frame)
        port_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(port_frame, text="Port:").pack(side=tk.LEFT, padx=(0, 5))
        self.target_port_entry = ttk.Entry(port_frame, textvariable=self.target_port_var, width=10)
        self.target_port_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(port_frame, text="(Default: 6000)").pack(side=tk.LEFT)
        
        # Connection buttons frame
        btn_frame = ttk.Frame(connection_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(
            btn_frame,
            text="Connection",
            command=self._test_connection,
            width=15
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            btn_frame,
            text="Refresh Local IP",
            command=self._update_local_ip,
            width=15
        ).pack(side=tk.LEFT)
        
        # Connection status
        self.connection_status_var = tk.StringVar(value="Not connected")
        ttk.Label(connection_frame, textvariable=self.connection_status_var, font=('TkDefaultFont', 9)).pack(anchor=tk.W, pady=(10, 0))
        
        # Transfer controls
        transfer_frame = ttk.LabelFrame(left_panel, text="Transfer Controls", padding=10)
        transfer_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons grid
        btn_grid = ttk.Frame(transfer_frame)
        btn_grid.pack(fill=tk.X)
        
        # Row 1
        row1 = ttk.Frame(btn_grid)
        row1.pack(fill=tk.X, pady=2)
        
        ttk.Button(
            row1,
            text="📄 Send File...",
            command=self._send_file_dialog,
            width=18
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            row1,
            text="📁 Send Folder...",
            command=self._send_folder_dialog,
            width=18
        ).pack(side=tk.LEFT)
        
        # Row 2
        row2 = ttk.Frame(btn_grid)
        row2.pack(fill=tk.X, pady=2)
        
        ttk.Button(
            row2,
            text="🎬 Send Video...",
            command=self._send_video_dialog,
            width=18
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            row2,
            text="📋 Sync Clipboard",
            command=self._sync_clipboard,
            width=18
        ).pack(side=tk.LEFT)
        
        # Row 3 - Clipboard sync controls
        row3 = ttk.Frame(btn_grid)
        row3.pack(fill=tk.X, pady=2)
        
        ttk.Button(
            row3,
            text="🔄 Start Sync",
            command=self._start_clipboard_sync,
            width=18
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            row3,
            text="⏸️ Stop Sync",
            command=self._stop_clipboard_sync,
            width=18
        ).pack(side=tk.LEFT)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(left_panel, text="Settings", padding=10)
        settings_frame.pack(fill=tk.X)
        
        ttk.Checkbutton(
            settings_frame,
            text="Auto-sync clipboard text",
            variable=self.auto_sync_var,
            command=self._toggle_auto_sync
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(
            settings_frame,
            text="Show notifications",
            variable=self.notify_var
        ).pack(anchor=tk.W, pady=2)
        
        # Right panel - Clipboard and logs
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=2)
        
        # Clipboard display
        clipboard_frame = ttk.LabelFrame(right_panel, text="Current Clipboard", padding=10)
        clipboard_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.clipboard_text = scrolledtext.ScrolledText(
            clipboard_frame,
            height=8,
            wrap=tk.WORD,
            font=('TkDefaultFont', 10)
        )
        self.clipboard_text.pack(fill=tk.BOTH, expand=True)
        self.clipboard_text.config(state='disabled')
        
        # Active transfers frame
        transfers_frame = ttk.LabelFrame(right_panel, text="Active Transfers", padding=10)
        transfers_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Transfers list
        self.transfers_text = scrolledtext.ScrolledText(
            transfers_frame,
            height=4,
            wrap=tk.WORD,
            font=('Courier', 9)
        )
        self.transfers_text.pack(fill=tk.BOTH, expand=True)
        self.transfers_text.config(state='disabled')
        
        # Log display
        log_frame = ttk.LabelFrame(right_panel, text="Activity Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=12,
            wrap=tk.WORD,
            font=('Courier', 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state='disabled')
        
        # Log controls
        log_controls = ttk.Frame(log_frame)
        log_controls.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(
            log_controls,
            text="Clear Log",
            command=self._clear_log,
            width=10
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            log_controls,
            text="Copy Log",
            command=self._copy_log,
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            log_controls,
            text="Open Received",
            command=self._open_received_folder,
            width=12
        ).pack(side=tk.RIGHT)
        
        # Status bar
        self.status_bar = ttk.Frame(self.root, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready | Enter target IP and port")
        status_label = ttk.Label(
            self.status_bar,
            textvariable=self.status_var,
            anchor=tk.W
        )
        status_label.pack(side=tk.LEFT, padx=5)
        
        # Circular progress indicator in status bar
        self.status_progress = CircularProgress(self.status_bar, size=20, thickness=3, color="#2196F3")
        self.status_progress.pack(side=tk.LEFT, padx=5)
        self.status_progress.start()
        
        self.connection_status_icon = tk.Label(self.status_bar, text="🔴", font=('TkDefaultFont', 10))
        self.connection_status_icon.pack(side=tk.RIGHT, padx=5)
        
    def _start_services(self):
        """Start all background services"""
        # Get port from entry or use default
        try:
            port = int(self.target_port_var.get())
        except ValueError:
            port = TCP_PORT
            self.target_port_var.set(str(port))
        
        self.receiver = ClipboardReceiver(
            port,
            self._clipboard_received,
            self._log_callback,
            self._progress_callback
        )
        
        self.clipboard_monitor = ClipboardMonitor(
            self._clipboard_changed,
            self._log_callback
        )
        
        self.sender = ClipboardSender(self._log_callback, self._progress_callback)
        self.clipboard_manager = ClipboardManager(self._log_callback)
        self.transfer_manager = FileTransferManager()
        
        # Start services
        self.receiver.start()
        time.sleep(1)
        self.clipboard_monitor.start()
        
    def _test_connection(self):
        """Test connection to target"""
        target_ip = self.target_ip_var.get().strip()
        target_port = self.target_port_var.get().strip()
        
        if not target_ip:
            messagebox.showwarning("No IP", "Please enter target IP address")
            return
        
        if not target_port:
            target_port = "6000"
            self.target_port_var.set(target_port)
        
        # Validate IP address
        if not NetworkUtils.validate_ip_address(target_ip):
            messagebox.showerror("Invalid IP", "Please enter a valid IP address (e.g., 192.168.1.100)")
            return
        
        # Validate port
        if not NetworkUtils.is_valid_port(target_port):
            messagebox.showerror("Invalid Port", "Please enter a valid port number (1-65535)")
            return
        
        target_port = int(target_port)
        
        try:
            self._log(f"🔧 Testing connection to {target_ip}:{target_port}")
            self.status_var.set(f"Testing connection to {target_ip}:{target_port}")
            self.connection_status_var.set("Testing connection...")
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            
            start_time = time.time()
            result = sock.connect_ex((target_ip, target_port))
            elapsed_time = (time.time() - start_time) * 1000
            
            sock.close()
            
            if result == 0:
                self._log(f"✅ Connection successful to {target_ip}:{target_port} ({elapsed_time:.0f}ms)")
                self.status_var.set(f"Connected to {target_ip}:{target_port}")
                self.connection_status_var.set(f"Connected to {target_ip}:{target_port}")
                self.connection_status_icon.config(text="🟢")
                # Save to cache
                self._save_cached_ip(target_ip, target_port)
                
                # Test clipboard sync
                threading.Thread(target=self._test_clipboard_sync, args=(target_ip, target_port), daemon=True).start()
            else:
                self._log(f"❌ Connection failed to {target_ip}:{target_port}")
                self.status_var.set(f"Connection failed to {target_ip}:{target_port}")
                self.connection_status_var.set(f"Connection failed")
                self.connection_status_icon.config(text="🔴")
                
        except Exception as e:
            self._log(f"❌ Connection error: {e}")
            self.status_var.set(f"Connection error: {e}")
            self.connection_status_var.set("Connection error")
            self.connection_status_icon.config(text="🔴")
    
    def _test_clipboard_sync(self, target_ip, target_port):
        """Test clipboard synchronization"""
        try:
            # Send a test message
            test_message = f"Clipboard sync test from {DEVICE_ID} at {time.strftime('%H:%M:%S')}"
            success = self.sender.send_text(target_ip, target_port, test_message)
            
            if success:
                self._log(f"✅ Clipboard sync test successful")
            else:
                self._log(f"⚠️ Clipboard sync test failed")
        except Exception as e:
            self._log(f"⚠️ Clipboard test error: {e}")
    
    def _sync_clipboard(self):
        """Manually sync clipboard to target"""
        if not self.auto_sync_var.get():
            return
        
        target_ip = self.target_ip_var.get().strip()
        target_port = self.target_port_var.get().strip()
        
        if not target_ip:
            messagebox.showwarning("No IP", "Please enter target IP address")
            return
        
        if not target_port:
            target_port = "6000"
            self.target_port_var.set(target_port)
        
        # Validate IP address
        if not NetworkUtils.validate_ip_address(target_ip):
            messagebox.showerror("Invalid IP", "Please enter a valid IP address")
            return
        
        # Validate port
        if not NetworkUtils.is_valid_port(target_port):
            messagebox.showerror("Invalid Port", "Please enter a valid port number")
            return
        
        target_port = int(target_port)
        
        try:
            # Get current clipboard text
            text = self.clipboard_manager.get_clipboard_text()
            if text and text.strip():
                success = self.sender.send_text(target_ip, target_port, text)
                if success:
                    self._log(f"📤 Sent clipboard to {target_ip}:{target_port}")
                    self._update_clipboard_display(text)
                else:
                    self._log(f"❌ Failed to send clipboard")
            else:
                self._log("📋 Clipboard is empty")
                
        except Exception as e:
            self._log(f"❌ Clipboard sync error: {e}")
    
    def _start_clipboard_sync(self):
        """Start automatic clipboard synchronization"""
        self.auto_sync_var.set(True)
        self._log("🔄 Auto-sync enabled")
        self.status_var.set("Auto-sync enabled")
    
    def _stop_clipboard_sync(self):
        """Stop automatic clipboard synchronization"""
        self.auto_sync_var.set(False)
        self._log("⏸️ Auto-sync disabled")
        self.status_var.set("Auto-sync disabled")
    
    def _clipboard_changed(self, clipboard_item):
        """Handle local clipboard changes"""
        if not self.auto_sync_var.get() or self.is_processing:
            return
        
        target_ip = self.target_ip_var.get().strip()
        target_port = self.target_port_var.get().strip()
        
        if not target_ip or not target_port:
            return
        
        # Validate IP and port
        if not NetworkUtils.validate_ip_address(target_ip):
            return
        
        try:
            target_port = int(target_port)
        except ValueError:
            return
        
        if clipboard_item.get('hash') == self.last_received_hash:
            return
        
        self._log(f"📋 Clipboard changed ({len(clipboard_item['data'])} chars)")
        self._update_clipboard_display(clipboard_item['data'])
        
        # Send to target
        threading.Thread(
            target=self._send_to_target,
            args=(target_ip, target_port, clipboard_item),
            daemon=True
        ).start()
    
    def _send_to_target(self, target_ip, target_port, clipboard_item):
        """Send clipboard item to target"""
        try:
            success = self.sender.send_text(target_ip, target_port, clipboard_item['data'])
            
            if success:
                self._log(f"📤 Sent to {target_ip}:{target_port}")
            else:
                self._log(f"❌ Failed to send to {target_ip}:{target_port}")
        except Exception as e:
            self._log(f"❌ Send error: {e}")
    
    def _clipboard_received(self, data):
        """Handle received clipboard data"""
        self.is_processing = True
        
        try:
            if data['type'] == 'text':
                text = data['data']
                source_ip = data.get('source_ip', 'Unknown')
                source_port = data.get('source_port', 'Unknown')
                
                text_hash = hashlib.md5(text.encode()).hexdigest()
                
                if text_hash == self.last_received_hash:
                    return
                    
                self.last_received_hash = text_hash
                
                success = self.clipboard_manager.set_clipboard_text(text)
                
                if success:
                    self._update_clipboard_display(text)
                    self._log(f"📥 Received text from {source_ip}:{source_port}")
                    
                    if self.notify_var.get():
                        self._show_notification(f"Clipboard updated from {source_ip}")
                else:
                    self._log("❌ Failed to set clipboard")
                    
            elif data['type'] == 'file':
                self._save_received_file(data)
                
        finally:
            self.is_processing = False
    
    
    def _save_received_file(self, data):
        """Save received file or folder with robust permission handling and fallbacks"""
        try:
            # Get file information from data
            filename = data.get('filename', f'received_{int(time.time())}')
            is_folder = data.get('is_folder', False)
            transfer_type = data.get('transfer_type', 'single')
            source_ip = data.get('source_ip', 'Unknown')
            
            # ===== STEP 1: DETERMINE SAVE LOCATION WITH FALLBACKS =====
            received_dir = None
            save_location_type = "Unknown"
            
            # Try multiple locations in order of preference
            possible_locations = [
                # 1. Primary location in user's home directory
                (Path.home() / "ClipboardReceived", "Home Directory"),
                # 2. Downloads folder (usually has write access)
                (Path.home() / "Downloads" / "ClipboardReceived", "Downloads"),
                # 3. Desktop (user-visible location)
                (Path.home() / "Desktop" / "ClipboardReceived", "Desktop"),
                # 4. Documents folder
                (Path.home() / "Documents" / "ClipboardReceived", "Documents"),
                # 5. Application Support (macOS convention)
                (Path.home() / "Library" / "Application Support" / "PyShineClipboard" / "Received", "Application Support"),
                # 6. Last resort: Temporary directory
                (Path(tempfile.gettempdir()) / "PyShineClipboard_Received", "Temporary")
            ]
            
            # Try each location until we find one we can write to
            for location, location_name in possible_locations:
                try:
                    # Create directory if it doesn't exist
                    location.mkdir(parents=True, exist_ok=True)
                    
                    # Test if we can write to this directory
                    test_file = location / ".write_test"
                    try:
                        test_file.touch(exist_ok=True)
                        # Write something to the file
                        with open(test_file, 'w') as f:
                            f.write("test")
                        test_file.unlink()  # Clean up
                        
                        # If we get here, we can write to this location
                        received_dir = location
                        
                        
                        save_location_type = location_name
                        self._log(f"✓ Using {location_name} for saving: {received_dir}")
                        break
                        
                    except (PermissionError, OSError) as e:
                        self._log(f"✗ Cannot write to {location_name}: {e}")
                        continue
                        
                except Exception as e:
                    self._log(f"✗ Failed to access {location_name}: {e}")
                    continue
            
            # If all automatic locations failed, ask the user
            if received_dir is None:
                self._log("⚠️ All automatic save locations failed. Asking user...")
                self.root.after(0, self._ask_for_save_location)
                # Wait a bit for user response (non-blocking in real app)
                time.sleep(1)
                
                # Check if user provided a location (you'll need to implement _get_user_save_location)
                user_dir = self._get_user_save_location()
                if user_dir:
                    received_dir = Path(user_dir)
                    save_location_type = "User Selected"
                    received_dir.mkdir(parents=True, exist_ok=True)
                    
                else:
                    self._log("❌ No save location available. File not saved.")
                    return
            
            # ===== STEP 2: SAVE THE ACTUAL FILE/FOLDER =====
            if is_folder:
                # It's a folder (ZIP file)
                original_name = data.get('original_folder_name', filename.replace('.zip', ''))
                
                # Save ZIP file
                zip_filename = f"{original_name}_{int(time.time())}.zip"
                zip_path = received_dir / zip_filename
                
                # Write ZIP data
                with open(zip_path, 'wb') as f:
                    f.write(data['data'])
                
                # Create extract path
                extract_path = received_dir / original_name
                
                # Avoid overwriting - add number if folder exists
                counter = 1
                while extract_path.exists():
                    extract_path = received_dir / f"{original_name}_{counter}"
                    counter += 1
                
                extract_path.mkdir(exist_ok=True)
                
                # Extract ZIP
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zipf:
                        # Get list of files for progress reporting
                        file_list = zipf.namelist()
                        total_files = len(file_list)
                        
                        self._log(f"📦 Extracting {total_files} files from {zip_filename}...")
                        
                        for i, file_info in enumerate(file_list, 1):
                            try:
                                zipf.extract(file_info, extract_path)
                            except Exception as e:
                                self._log(f"⚠️ Failed to extract {file_info}: {e}")
                                continue
                            
                            # Log progress for large archives
                            if total_files > 10 and i % max(1, total_files // 10) == 0:
                                progress = (i / total_files) * 100
                                self._log(f"   Extracting... {progress:.0f}% ({i}/{total_files})")
                    
                    self._log(f"✅ Extraction complete: {extract_path}")
                    
                except zipfile.BadZipFile:
                    self._log(f"❌ Error: {zip_filename} is not a valid ZIP file")
                    # Try to save the raw file anyway
                    raw_path = received_dir / f"{original_name}.zip"
                    shutil.copy2(zip_path, raw_path)
                    extract_path = raw_path
                    self._log(f"💾 Saved as raw file instead: {raw_path}")
                
                # Clean up ZIP file after successful extraction
                try:
                    if zip_path.exists():
                        zip_path.unlink()
                        self._log(f"🧹 Cleaned up ZIP file: {zip_filename}")
                except:
                    pass  # Don't worry if we can't delete the zip
                
                final_path = extract_path
                self._log(f"📁 Folder saved: {original_name} → {save_location_type}")
                
            else:
                # Regular file
                # Handle duplicate filenames
                filepath = received_dir / filename
                counter = 1
                name_parts = filename.rsplit('.', 1)
                
                while filepath.exists():
                    if len(name_parts) == 2:
                        new_filename = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                    else:
                        new_filename = f"{filename}_{counter}"
                    filepath = received_dir / new_filename
                    counter += 1
                    if counter > 100:  # Safety limit
                        filepath = received_dir / f"{int(time.time())}_{filename}"
                        break
                
                # Save the file
                with open(filepath, 'wb') as f:
                    f.write(data['data'])
                
                final_path = filepath
                file_size = len(data['data'])
                size_str = FileTransferManager.format_size(file_size)
                self._log(f"📄 File saved: {filename} ({size_str}) → {save_location_type}")
            
            # ===== STEP 3: POST-SAVE ACTIONS =====
            # Add to recent transfers list
            self.received_files.append({
                'path': str(final_path),
                'name': final_path.name,
                'is_folder': is_folder,
                'timestamp': time.time(),
                'size': len(data['data']),
                'location': save_location_type,
                'source': source_ip
            })
            
            # Update transfers list display
            self._update_transfers_list()
            
            # Show notification if enabled
            if self.notify_var.get():
                if is_folder:
                    self._show_notification(f"📁 Folder received from {source_ip} " + f"{final_path.name}\nSaved to: {save_location_type}")
                else:
                    self._show_notification(f"📄 File received from {source_ip} " + f"{final_path.name}\nSaved to: {save_location_type}")
            
            # ===== STEP 4: ASK USER WHAT TO DO =====
            def ask_user_action():
                if is_folder:
                    response = messagebox.askyesnocancel(
                        "Folder Received",
                        f"Received folder: {final_path.name}\n\n"
                        f"Location: {save_location_type}\n"
                        f"From: {source_ip}\n\n"
                        "What would you like to do?",
                        detail="Yes: Open folder\nNo: Show in Finder\nCancel: Do nothing"
                    )
                    
                    if response is True:  # Yes - Open folder
                        error = self.transfer_manager.open_folder(final_path)
                        if error:
                            self._log(f"❌ Failed to open folder: {error}")
                    elif response is False:  # No - Show in Finder
                        try:
                            if platform.system() == "Darwin":
                                subprocess.run(['open', '-R', str(final_path)])
                            elif platform.system() == "Windows":
                                subprocess.run(['explorer', '/select,', str(final_path)])
                            else:
                                subprocess.run(['xdg-open', str(final_path.parent)])
                        except Exception as e:
                            self._log(f"❌ Failed to show in file manager: {e}")
                    
                else:  # Regular file
                    response = messagebox.askyesnocancel(
                        "File Received",
                        f"Received file: {final_path.name}\n\n"
                        f"Location: {save_location_type}\n"
                        f"From: {source_ip}\n"
                        f"Size: {FileTransferManager.format_size(len(data['data']))}\n\n"
                        "Open the file?",
                        detail="Yes: Open file\nNo: Show in Finder\nCancel: Do nothing"
                    )
                    
                    if response is True:  # Yes - Open file
                        error = self.transfer_manager.open_file(final_path)
                        if error:
                            self._log(f"❌ Failed to open file: {error}")
                    elif response is False:  # No - Show in Finder
                        try:
                            if platform.system() == "Darwin":
                                subprocess.run(['open', '-R', str(final_path)])
                            elif platform.system() == "Windows":
                                subprocess.run(['explorer', '/select,', str(final_path)])
                            else:
                                subprocess.run(['xdg-open', str(final_path.parent)])
                        except Exception as e:
                            self._log(f"❌ Failed to show in file manager: {e}")
            
            # Ask user in main thread
            self.root.after(100, ask_user_action)
            
            # Log success
            self._log(f"✅ Successfully saved to: {final_path}")
            # Store the parent directory for future "Open Received" clicks
            self.primary_received_dir = final_path.parent  # <-- ADD THIS LINE
            
        except Exception as e:
            self._log(f"❌ Critical error saving file: {e}")
            import traceback
            traceback.print_exc()
            
            # Last resort: save to desktop with timestamp
            try:
                emergency_path = Path.home() / "Desktop" / f"CLIPBOARD_EMERGENCY_{int(time.time())}.dat"
                with open(emergency_path, 'wb') as f:
                    f.write(data['data'])
                self._log(f"🚨 Emergency save to Desktop: {emergency_path}")
            except Exception as final_e:
                self._log(f"💥 All save attempts failed: {final_e}")

    def _ask_for_save_location(self):
        """Ask user for save location (called from main thread)"""
        save_path = filedialog.askdirectory(
            title="Select folder to save received files",
            initialdir=str(Path.home())
        )
        if save_path:
            self._user_save_location = save_path
            self._log(f"📁 User selected save location: {save_path}")
        else:
            self._user_save_location = None

    def _get_user_save_location(self):
        """Get the user-selected save location"""
        return getattr(self, '_user_save_location', None)
    
    
    def __save_received_file(self, data):
        """Save received file or folder"""
        try:
            # Create received directory
            received_dir = Path.home() / "ClipboardReceived"
            received_dir.mkdir(exist_ok=True)
            
            filename = data.get('filename', f'received_{int(time.time())}')
            is_folder = data.get('is_folder', False)
            transfer_type = data.get('transfer_type', 'single')
            
            if is_folder:
                # It's a folder (ZIP file)
                original_name = data.get('original_folder_name', filename.replace('.zip', ''))
                
                # Save ZIP file
                zip_path = received_dir / filename
                with open(zip_path, 'wb') as f:
                    f.write(data['data'])
                
                # Extract ZIP
                extract_path = received_dir / original_name
                extract_path.mkdir(exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    zipf.extractall(extract_path)
                
                # Clean up ZIP
                try:
                    os.remove(zip_path)
                except:
                    pass
                
                final_path = extract_path
                self._log(f"📁 Folder received: {original_name}")
                
            else:
                # Regular file
                filepath = received_dir / filename
                with open(filepath, 'wb') as f:
                    f.write(data['data'])
                
                final_path = filepath
                self._log(f"📄 File received: {filename}")
            
            # Add to recent transfers list
            self.received_files.append({
                'path': str(final_path),
                'name': final_path.name,
                'is_folder': is_folder,
                'timestamp': time.time(),
                'size': len(data['data'])
            })
            
            # Update files list
            self._update_transfers_list()
            
            source_ip = data.get('source_ip', 'Unknown')
            
            if self.notify_var.get():
                if is_folder:
                    self._show_notification(f"Folder received from {source_ip}: {final_path.name}")
                else:
                    self._show_notification(f"File received from {source_ip}: {final_path.name}")
            
            # Ask to open
            if is_folder:
                if messagebox.askyesno("Folder Received", f"Received folder: {final_path.name}\n\nOpen folder?"):
                    self.transfer_manager.open_folder(final_path)
            else:
                if messagebox.askyesno("File Received", f"Received file: {final_path.name}\n\nOpen file?"):
                    self.transfer_manager.open_file(final_path)
                
        except Exception as e:
            self._log(f"❌ Error saving file: {e}")
    
    def _send_file_dialog(self):
        """Send any file to target"""
        target_ip = self.target_ip_var.get().strip()
        target_port = self.target_port_var.get().strip()
        
        if not target_ip:
            messagebox.showwarning("No Target", "Please enter target IP address")
            return
        
        if not target_port:
            target_port = "6000"
            self.target_port_var.set(target_port)
        
        # Validate IP and port
        if not NetworkUtils.validate_ip_address(target_ip):
            messagebox.showerror("Invalid IP", "Please enter a valid IP address")
            return
        
        if not NetworkUtils.is_valid_port(target_port):
            messagebox.showerror("Invalid Port", "Please enter a valid port number")
            return
        
        target_port = int(target_port)
        
        filepaths = filedialog.askopenfilenames(
            title="Select files to send",
            filetypes=[
                ("All files", "*.*"),
                ("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("Images", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff *.webp"),
                ("Documents", "*.pdf *.doc *.docx *.txt *.rtf *.odt"),
                ("Audio", "*.mp3 *.wav *.flac *.aac *.ogg *.m4a"),
                ("Archives", "*.zip *.rar *.7z *.tar *.gz"),
                ("Code", "*.py *.js *.html *.css *.java *.cpp *.c *.php"),
            ]
        )
        
        if not filepaths:
            return
        
        self._send_files(target_ip, target_port, filepaths)
    
    def _send_folder_dialog(self):
        """Send folder to target"""
        target_ip = self.target_ip_var.get().strip()
        target_port = self.target_port_var.get().strip()
        
        if not target_ip:
            messagebox.showwarning("No Target", "Please enter target IP address")
            return
        
        if not target_port:
            target_port = "6000"
            self.target_port_var.set(target_port)
        
        # Validate IP and port
        if not NetworkUtils.validate_ip_address(target_ip):
            messagebox.showerror("Invalid IP", "Please enter a valid IP address")
            return
        
        if not NetworkUtils.is_valid_port(target_port):
            messagebox.showerror("Invalid Port", "Please enter a valid port number")
            return
        
        target_port = int(target_port)
        
        folder_path = filedialog.askdirectory(title="Select folder to send")
        
        if not folder_path:
            return
        
        # Get folder info
        folder_info, error = self.transfer_manager.get_file_info(folder_path)
        if error:
            messagebox.showerror("Error", f"Invalid folder: {error}")
            return
        
        # Confirm transfer
        file_count = folder_info.get('file_count', 0)
        total_size = folder_info.get('total_size', 0)
        
        confirm = messagebox.askyesno(
            "Confirm Folder Transfer",
            f"Folder: {Path(folder_path).name}\n"
            f"Files: {file_count}\n"
            f"Total size: {self.transfer_manager.format_size(total_size)}\n\n"
            f"Send to {target_ip}:{target_port}?"
        )
        
        if confirm:
            self._send_folder(target_ip, target_port, folder_path)
    
    def _send_video_dialog(self):
        """Send video files"""
        target_ip = self.target_ip_var.get().strip()
        target_port = self.target_port_var.get().strip()
        
        if not target_ip:
            messagebox.showwarning("No Target", "Please enter target IP address")
            return
        
        if not target_port:
            target_port = "6000"
            self.target_port_var.set(target_port)
        
        # Validate IP and port
        if not NetworkUtils.validate_ip_address(target_ip):
            messagebox.showerror("Invalid IP", "Please enter a valid IP address")
            return
        
        if not NetworkUtils.is_valid_port(target_port):
            messagebox.showerror("Invalid Port", "Please enter a valid port number")
            return
        
        target_port = int(target_port)
        # Save to cache
        self._save_cached_ip(target_ip, target_port)
        
        filepaths = filedialog.askopenfilenames(
            title="Select video files to send",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.mpg *.mpeg"),
                ("All files", "*.*"),
            ]
        )
        
        if not filepaths:
            return
        
        self._send_files(target_ip, target_port, filepaths)
    
    def _send_files(self, target_ip, target_port, filepaths):
        """Send multiple files"""
        for filepath in filepaths:
            file_info, error = self.transfer_manager.get_file_info(filepath)
            if error:
                self._log(f"❌ Error with {Path(filepath).name}: {error}")
                continue
            
            # Show progress dialog for large files
            if file_info['size'] > 10 * 1024 * 1024:  # > 10MB
                transfer_id = f"send_{target_ip}_{int(time.time())}"
                
                # Create circular progress dialog
                dialog = CircularProgressDialog(
                    self.root,
                    Path(filepath).name,
                    file_info['size'],
                    'send',
                    f"{target_ip}:{target_port}"
                )
                
                # Store dialog reference
                self.progress_dialogs[transfer_id] = dialog
                
                # Start transfer in background thread
                threading.Thread(
                    target=self._send_file_with_progress,
                    args=(target_ip, target_port, filepath, False, transfer_id, dialog),
                    daemon=True
                ).start()
            else:
                # Small file, send without progress dialog
                self._log(f"📤 Sending {Path(filepath).name} ({self.transfer_manager.format_size(file_info['size'])}) to {target_ip}:{target_port}...")
                
                success, error = self.sender.send_file(target_ip, target_port, filepath)
                
                if success:
                    self._log(f"✅ Sent: {Path(filepath).name}")
                else:
                    self._log(f"❌ Failed to send {Path(filepath).name}: {error}")
    
    def _send_folder(self, target_ip, target_port, folder_path):
        """Send folder as ZIP"""
        # Get folder info
        folder_info, error = self.transfer_manager.get_file_info(folder_path)
        if error:
            messagebox.showerror("Error", f"Invalid folder: {error}")
            return
        
        # Create progress dialog
        transfer_id = f"send_folder_{target_ip}_{int(time.time())}"
        dialog = CircularProgressDialog(
            self.root,
            Path(folder_path).name,
            folder_info.get('total_size', 0),
            'send',
            f"{target_ip}:{target_port}"
        )
        
        # Store dialog reference
        self.progress_dialogs[transfer_id] = dialog
        
        # Start transfer in background thread
        threading.Thread(
            target=self._send_folder_with_progress,
            args=(target_ip, target_port, folder_path, transfer_id, dialog),
            daemon=True
        ).start()
    
    def _send_file_with_progress(self, target_ip, target_port, filepath, is_folder, transfer_id, dialog):
        """Send file with circular progress dialog"""
        try:
            success, error = self.sender.send_file(target_ip, target_port, filepath, is_folder, transfer_id)
            
            # Update dialog on completion
            self.root.after(0, lambda: dialog.complete(success, error))
            
            if success:
                self._log(f"✅ Sent: {Path(filepath).name}")
            else:
                self._log(f"❌ Failed to send {Path(filepath).name}: {error}")
        except Exception as e:
            self.root.after(0, lambda: dialog.complete(False, str(e)))
            self._log(f"❌ Error sending {Path(filepath).name}: {e}")
        finally:
            # Remove dialog reference after a delay
            self.root.after(3000, lambda: self._remove_dialog(transfer_id))
    
    def _send_folder_with_progress(self, target_ip, target_port, folder_path, transfer_id, dialog):
        """Send folder with circular progress dialog"""
        try:
            success, error = self.sender.send_folder_as_zip(target_ip, target_port, folder_path, transfer_id)
            
            # Update dialog on completion
            self.root.after(0, lambda: dialog.complete(success, error))
            
            if success:
                self._log(f"✅ Folder sent: {Path(folder_path).name}")
            else:
                self._log(f"❌ Failed to send folder: {error}")
        except Exception as e:
            self.root.after(0, lambda: dialog.complete(False, str(e)))
            self._log(f"❌ Error sending folder: {e}")
        finally:
            # Remove dialog reference after a delay
            self.root.after(3000, lambda: self._remove_dialog(transfer_id))
    
    def _remove_dialog(self, transfer_id):
        """Remove progress dialog reference"""
        if transfer_id in self.progress_dialogs:
            del self.progress_dialogs[transfer_id]
    
    def _toggle_auto_sync(self):
        """Toggle auto-sync"""
        if self.auto_sync_var.get():
            self._log("Auto-sync enabled")
        else:
            self._log("Auto-sync disabled")
    
    def _show_notification(self, message):
        """Show notification"""
        if not self.notify_var.get():
            return
        
        self._log(f"📢 {message}")
    
    def __open_received_folder(self):
        """Open received files folder"""
        received_dir = Path.home() / "ClipboardReceived"
        received_dir.mkdir(exist_ok=True)
        
        error = self.transfer_manager.open_folder(received_dir)
        if error:
            self._log(f"❌ Failed to open folder: {error}")
    
    def _open_received_folder(self):
        """Open the folder where files are actually being saved"""
        try:
            # 1. Try to use the directory from the most recent save
            if hasattr(self, 'primary_received_dir') and self.primary_received_dir:
                received_dir = self.primary_received_dir
                self._log(f"📂 Opening from recent save location: {received_dir}")
            
            # 2. Fallback: check received_files list for last saved file
            elif self.received_files:
                last_file_path = Path(self.received_files[-1]['path'])
                received_dir = last_file_path.parent
                self._log(f"📂 Opening from last received file: {received_dir}")
                # Update primary for next time
                self.primary_received_dir = received_dir
            
            # 3. Ultimate fallback: default location
            else:
                received_dir = Path.home() / "ClipboardReceived"
                received_dir.mkdir(exist_ok=True)
                self._log(f"📂 Opening default folder (no files received yet): {received_dir}")
                self.primary_received_dir = received_dir
            
            # Ensure the directory exists
            if not received_dir.exists():
                received_dir.mkdir(parents=True, exist_ok=True)
                self._log(f"📁 Created missing directory: {received_dir}")
            
            # Open it
            error = self.transfer_manager.open_folder(received_dir)
            if error:
                self._log(f"❌ Failed to open folder {received_dir}: {error}")
                
        except Exception as e:
            self._log(f"❌ Error in _open_received_folder: {e}")
            # Emergency fallback
            try:
                fallback = Path.home() / "ClipboardReceived"
                fallback.mkdir(exist_ok=True)
                self.transfer_manager.open_folder(fallback)
            except:
                pass  # Nothing more we can do
    
    def _update_clipboard_display(self, text):
        """Update clipboard display"""
        self.root.after(0, lambda: self._update_clipboard_display_threadsafe(text))
    
    def _update_clipboard_display_threadsafe(self, text):
        """Thread-safe UI update"""
        self.clipboard_text.config(state='normal')
        self.clipboard_text.delete(1.0, tk.END)
        self.clipboard_text.insert(1.0, text)  # No truncation
        self.clipboard_text.config(state='disabled')
    
    def _update_transfers_list(self):
        """Update active transfers list"""
        self.root.after(0, self._update_transfers_list_threadsafe)
    
    def _update_transfers_list_threadsafe(self):
        """Thread-safe transfers list update"""
        self.transfers_text.config(state='normal')
        self.transfers_text.delete(1.0, tk.END)
        
        # Get active transfers from progress manager
        active_transfers = self.progress_manager.get_active_transfers()
        
        if not active_transfers:
            self.transfers_text.insert(1.0, "No active transfers")
        else:
            for transfer_id, transfer in active_transfers:
                filename = transfer['filename']
                current_size = transfer['current_size']
                total_size = transfer['total_size']
                transfer_type = "Sending" if transfer['transfer_type'] == 'send' else "Receiving"
                
                # Show size instead of percentage
                current_str = FileTransferManager.format_size(current_size)
                total_str = FileTransferManager.format_size(total_size)
                
                line = f"{transfer_type}: {filename} - {current_str} / {total_str}\n"
                self.transfers_text.insert(tk.END, line)
        
        self.transfers_text.config(state='disabled')
    
    def _log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_queue.put(log_entry)
    
    def _log_callback(self, message):
        """Callback for logging from threads"""
        self._log(message)
    
    def _progress_callback(self, event_type, data):
        """Handle progress updates from transfers"""
        # Put in queue for thread-safe GUI update
        self.progress_queue.put((event_type, data))
    
    def _update_gui_loop(self):
        """Update GUI periodically"""
        # Process log queue
        try:
            while True:
                log_entry = self.log_queue.get_nowait()
                self.log_text.config(state='normal')
                self.log_text.insert(tk.END, log_entry)
                self.log_text.see(tk.END)
                self.log_text.config(state='disabled')
                
                if self.log_text.index('end-1c').split('.')[0] > '500':
                    self.log_text.delete(1.0, '2.0')
                    
        except queue.Empty:
            pass
        
        # Process progress queue
        try:
            while True:
                event_type, data = self.progress_queue.get_nowait()
                
                if event_type == 'transfer_started':
                    transfer_id = data['transfer_id']
                    self.active_transfers[transfer_id] = data
                    
                    # Update status bar
                    self.status_var.set(f"{'Sending' if data['transfer_type'] == 'send' else 'Receiving'}: {data['filename']}")
                    
                elif event_type == 'progress_updated':
                    transfer_id = data['transfer_id']
                    
                    # Update active transfer
                    if transfer_id in self.active_transfers:
                        self.active_transfers[transfer_id].update(data)
                    
                    # Update progress dialog if exists
                    if transfer_id in self.progress_dialogs:
                        dialog = self.progress_dialogs[transfer_id]
                        if not dialog.is_cancelled():
                            self.root.after(0, lambda: dialog.update_progress(
                                data['percentage'],
                                data['current_size'],
                                data['total_size']
                            ))
                    
                elif event_type == 'transfer_completed':
                    transfer_id = data['transfer_id']
                    
                    # Remove from active transfers
                    if transfer_id in self.active_transfers:
                        del self.active_transfers[transfer_id]
                    
                    # Update status bar
                    self.status_var.set("Ready")
                    
                    # Log completion
                    if data['success']:
                        time_str = f"{data['total_time']:.1f}s"
                        size_str = self.transfer_manager.format_size(data['total_size'])
                        self._log(f"✅ Transfer completed: {data['filename']} ({size_str}) in {time_str}")
                    else:
                        self._log(f"❌ Transfer failed: {data['filename']} - {data['error_msg']}")
                
                # Update transfers list
                self._update_transfers_list()
                    
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(100, self._update_gui_loop)
    
    def _clear_log(self):
        """Clear log"""
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')
    
    def _copy_log(self):
        """Copy log to clipboard"""
        try:
            log_content = self.log_text.get(1.0, tk.END)
            pyperclip.copy(log_content)
            self._log("Log copied to clipboard")
        except:
            pass
    
    # ====== CRITICAL FIX: Enhanced on_closing method ======
    def on_closing(self):
        """Handle application closing - FIXED for proper termination"""
        try:
            self._log("[SHUTDOWN] Application shutdown initiated...")
            
            # Ask for confirmation
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                self._log("[SHUTDOWN] Shutting down services...")
                
                # Stop all services
                self.receiver.running = False
                self.clipboard_monitor.running = False
                
                # Call stop methods
                try:
                    self.receiver.stop()
                except:
                    pass
                    
                try:
                    self.clipboard_monitor.stop()
                except:
                    pass
                
                # Wait a moment for threads to stop
                self.root.after(100, self._force_shutdown)
                
        except Exception as e:
            self._log(f"[SHUTDOWN] Error during shutdown: {e}")
            # Force shutdown anyway
            self._force_shutdown()
    
    def _force_shutdown(self):
        """Force shutdown of the application"""
        try:
            # Destroy all windows
            self.root.quit()
            self.root.destroy()
            
            # Force exit - this is CRITICAL for PyInstaller
            self._log("[SHUTDOWN] Forcing application exit...")
            
            # Kill the process
            if hasattr(os, '_exit'):
                os._exit(0)  # Immediate termination, bypassing Python cleanup
            else:
                sys.exit(0)  # Fallback
                
        except:
            # Ultimate fallback
            import ctypes
            ctypes.windll.user32.PostQuitMessage(0) if platform.system() == "Windows" else None
            os._exit(0)

# ------------------ Resource Path Helper ------------------
def get_resource_path(relative_path):
    """Get the correct path for resources whether running as script or PyInstaller executable"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Running as normal Python script
        base_path = os.path.abspath(".")
    
    path = os.path.join(base_path, relative_path)
    
    # Check if file exists, if not try other common locations
    if not os.path.exists(path):
        # Try parent directory
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)
        if os.path.exists(alt_path):
            return alt_path
        
        # Try current working directory
        alt_path = os.path.join(os.getcwd(), relative_path)
        if os.path.exists(alt_path):
            return alt_path
        
        # Try the actual executable directory (for PyInstaller)
        if hasattr(sys, 'frozen'):
            exe_dir = os.path.dirname(sys.executable)
            alt_path = os.path.join(exe_dir, relative_path)
            if os.path.exists(alt_path):
                return alt_path
    
    return path

def load_application_icon(root):
    """Simple wrapper for backward compatibility"""
    return setup_application_icon(root)

def _load_application_icon(root):
    """Load application icon with multiple fallbacks, including PyInstaller bundle"""
    # First, try the icon that PyInstaller might have embedded
    try:
        if hasattr(sys, '_MEIPASS'):
            # Running as PyInstaller bundle
            bundle_dir = Path(sys._MEIPASS)
            icon_path = bundle_dir / "icon_clean.png"
            if icon_path.exists():
                icon = tk.PhotoImage(file=str(icon_path))
                root.iconphoto(True, icon)
                root._icon = icon  # Keep reference
                print(f"Icon loaded from bundle: {icon_path}")
                return True
    except Exception as e:
        print(f"Failed to load icon from bundle: {e}")
    
    # Try multiple icon formats and locations
    icon_formats = [
        "icon_clean.png",
        "icon.png",
        "logo.png",
        "app.ico",
        "logo.ico",
        "icon.ico",
    ]
    
    # Check multiple possible locations
    search_paths = [
        Path.cwd(),  # Current working directory
        Path(sys.executable).parent if hasattr(sys, 'frozen') else Path(__file__).parent,  # Executable directory
        Path.home() / "Downloads",
        Path.home() / "Desktop",
    ]
    
    for icon_file in icon_formats:
        for search_path in search_paths:
            try:
                icon_path = search_path / icon_file
                if icon_path.exists():
                    print(f"Trying icon: {icon_path}")
                    
                    if icon_file.endswith('.ico'):
                        # For Windows .ico files
                        if platform.system() == "Windows":
                            root.iconbitmap(str(icon_path))
                            return True
                        else:
                            # On non-Windows, try to load as image
                            try:
                                from PIL import ImageTk, Image
                                img = Image.open(icon_path)
                                photo = ImageTk.PhotoImage(img)
                                root.iconphoto(True, photo)
                                root._icon = photo
                                return True
                            except ImportError:
                                pass
                    else:
                        # For PNG/GIF files
                        try:
                            icon = tk.PhotoImage(file=str(icon_path))
                            root.iconphoto(True, icon)
                            root._icon = icon
                            print(f"Icon loaded successfully: {icon_path}")
                            return True
                        except Exception as e:
                            print(f"Failed to load {icon_path}: {e}")
                            continue
            except Exception as e:
                print(f"Error checking {icon_file} in {search_path}: {e}")
                continue
    
    print("No icon found, using default Tkinter icon")
    return False

def _load_application_icon(root):
    """Load application icon with multiple fallbacks"""
    icon_formats = [
        "icon_clean.png",
        "icon.png",
        "logo.png",
        "app.ico",
        "logo.ico",
        "icon.ico",
    ]
    
    for icon_file in icon_formats:
        try:
            icon_path = get_resource_path(icon_file)
            
            if os.path.exists(icon_path):
                if icon_file.endswith('.ico'):
                    # For Windows .ico files
                    if platform.system() == "Windows":
                        root.iconbitmap(icon_path)
                        return True
                    else:
                        # On non-Windows, try to load as image
                        try:
                            from PIL import ImageTk, Image
                            img = Image.open(icon_path)
                            photo = ImageTk.PhotoImage(img)
                            root.iconphoto(True, photo)
                            # Keep reference to prevent garbage collection
                            root._icon = photo
                            return True
                        except ImportError:
                            pass
                else:
                    # For PNG/GIF files
                    try:
                        icon = tk.PhotoImage(file=icon_path)
                        root.iconphoto(True, icon)
                        # Keep reference to prevent garbage collection
                        root._icon = icon
                        return True
                    except:
                        continue
        except Exception:
            continue
    
    return False

# ------------------ Installation ------------------
def install_dependencies():
    """Install required dependencies"""
    required = ['pyperclip']
    
    if platform.system() == "Darwin":
        required.append('pyclip')  # Better for macOS
        required.append('netifaces')
    elif platform.system() == "Windows":
        # required.append('pywin32')
        required.append('netifaces')
    else:  # Linux
        required.append('xclip')  # For Linux clipboard
        required.append('netifaces')
    
    missing = []
    for module in required:
        try:
            if module == 'xclip':
                # Check if xclip command exists on Linux
                if platform.system() == "Linux":
                    result = subprocess.run(['which', 'xclip'], capture_output=True)
                    if result.returncode != 0:
                        missing.append('xclip (system package)')
            else:
                __import__(module.replace('-', '_'))
        except ImportError:
            missing.append(module)
    
    if missing:
        print("Installing missing dependencies...")
        
        try:
            import subprocess
            pip_install = [m for m in missing if '(system package)' not in m]
            if pip_install:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + pip_install)
            
            # Print system package instructions
            for m in missing:
                if '(system package)' in m:
                    print(f"\nFor {m}, install system package:")
                    if platform.system() == "Linux":
                        print(f"  sudo apt-get install {m.split(' ')[0]}")
        except Exception as e:
            print(f"Error installing dependencies: {e}")
            print("\nPlease install manually:")
            print("  pip install", " ".join([m for m in missing if '(system package)' not in m]))

def check_firewall():
    """Check firewall settings"""
    print("\n" + "="*60)
    print("FIREWALL CONFIGURATION")
    print("="*60)
    
    if platform.system() == "Windows":
        print("1. Allow Python through Windows Defender Firewall")
        print("2. Allow port 6000 (TCP)")
        print("3. Run as Administrator if needed")
    elif platform.system() == "Darwin":
        print("1. System Settings > Privacy & Security > Firewall")
        print("2. Click 'Firewall Options'")
        print("3. Allow Python/terminal")
        print("\nClipboard Access:")
        print("4. System Settings > Privacy & Security > Privacy")
        print("5. Select 'Automation' or 'Accessibility'")
        print("6. Add Terminal/iTerm and grant clipboard access")
    else:
        print("If using ufw:")
        print("  sudo ufw allow 6000/tcp")
    
    print("="*60 + "\n")

# ------------------ Main ------------------
def main():
    """Main entry point"""
    print("="*60)
    print("PyShine Clipboard Pro - v1.0")
    print("="*60)
    print("Features:")
    print("• DIRECT CONNECTION BETWEEN TWO PCs")
    print("• Enter target IP and port (default: 6000)")
    print("• Auto-sync clipboard text")
    print("• Send ANY file type (videos, images, documents, etc.)")
    print("• Send complete folders with structure")
    print("• Circular progress indicators")
    print("• Cross-platform (Windows, macOS, Linux)")
    print("="*60)
    print("\nHOW TO USE:")
    print("1. Enter target IP address")
    print("2. Enter target port (default: 6000)")
    print("3. Click 'Connection' to verify")
    print("4. Use 'Send File/Folder' to transfer files")
    print("5. Enable 'Auto-sync' for automatic clipboard sharing")
    print("="*60)
    
    # Install dependencies
    try:
        install_dependencies()
    except Exception as e:
        print(f"Warning: Could not install all dependencies: {e}")
    
    # Check firewall
    check_firewall()
    
    # Initialize mimetypes
    mimetypes.init()
    
    # Create and run application
    root = tk.Tk()
    
    # Load icon dynamically
    load_application_icon(root)

    app = PyShineClipboardApp(root)
    
    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Start main loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nShutting down...")
        # Force exit on Ctrl+C
        os._exit(0)
    except Exception as e:
        print(f"Error: {e}")
        os._exit(1)

if __name__ == "__main__":
    main()

```

## Privacy First

- No cloud servers
- No accounts
- No telemetry
- Your data stays on your network

---

## Support & Updates

- Website: https://www.pyshine.com

---

Made with ❤️ by **PyShine**
