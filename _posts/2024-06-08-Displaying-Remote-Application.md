---
description: In this tutorial, we'll walk you through the steps to display graphical applications running on a remote server on your local machine. This is especially use...
featured-img: 26072022-python-logo
keywords:
- X11 forwarding
- SSH
- XQuartz
- remote GUI applications
- Matplotlib
- Ubuntu
- macOS
layout: post
mathjax: true
tags:
- X11
- SSH
- XQuartz
- remote applications
- GUI
- Matplotlib
- Ubuntu
- macOS
title: Displaying Remote Application Windows Locally Using SSH...
---


In this tutorial, we'll walk you through the steps to display graphical applications running on a remote server on your local machine. This is especially useful for visualizing plots or running GUI-based applications from a remote server.

# Prerequisites
## Remote Server: 
Ubuntu 22.04 or similar.
### Local Machine: 
macOS with XQuartz installed.
### SSH Access: 
Ensure you can SSH into the remote server.

## Step 1: Install XQuartz on macOS
XQuartz is an open-source version of the X.Org X server that runs on macOS.

### Download XQuartz:

Go to the XQuartz website and download the latest version.
### Install XQuartz:

Open the downloaded .dmg file and follow the installation instructions.
### Start XQuartz:

Open XQuartz from the Applications folder.


## Step 2: Configure SSH for X11 Forwarding
## On the Remote Server
Edit the SSH Configuration File:

Open the SSH configuration file with a text editor, such as nano:

```
sudo nano /etc/ssh/sshd_config
```

## Uncomment and Set Values:

Ensure the following lines are set (uncomment if necessary):
```
X11Forwarding yes
X11UseLocalhost yes

```

## Save and Exit:

Save the changes and exit the text editor (Ctrl+O to save in nano, then Ctrl+X to exit).
## Restart SSH Service:

Restart the SSH service to apply the changes:

```
sudo systemctl restart ssh

```

## Step 3: SSH with X11 Forwarding from macOS
Set DISPLAY Variable on macOS:

In a terminal on your Mac, set the DISPLAY variable:
```
export DISPLAY=:0

```

## SSH into the Remote Server:

Use the -Y flag to enable trusted X11 forwarding:

```
ssh -Y user@remote_server_ip

```

Replace user with your username on the remote server and remote_server_ip with the IP address of your remote server.

## Step 4: Test X11 Forwarding
## Run a GUI Application:
After logging into the remote server, try running an X11 application, such as `xclock`:
```
xclock
```

If everything is configured correctly, the xclock window should appear on your local machine.

## Troubleshooting
If you encounter the "Can't open display" error or other issues, try the following steps:

## Ensure XQuartz is Allowing Connections:

In XQuartz, go to Preferences > Security and ensure "Allow connections from network clients" is checked.
Check DISPLAY Variable on the Remote Server:

After SSHing into the remote server, ensure the DISPLAY variable is set correctly:

```
echo $DISPLAY
```
The output should be something like localhost:10.0. or something like that with a port number .
```
Restart XQuartz:
```


```
import matplotlib.pyplot as plt
plt.plot([1,2,3])
plt.show()
```


Sometimes, simply restarting XQuartz can resolve connection issues.

## Conclusion
By following these steps, you can run graphical applications on a remote server and display them on your local machine using X11 forwarding. This setup is particularly useful for remote development, running graphical applications, and visualizing data plots from a remote server. Thanks





