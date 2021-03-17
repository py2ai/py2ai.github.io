---
layout: post
title: How to easily install OpenCv in Raspberry Pi boards
categories: [Socket Programming Series]
mathjax: true
featured-img: None
summary: This tutorial is about installing Open CV library in Raspberry Pi 
---

Hi! friends, Raspberry Pi is not only a small but also comes with a lot of powerful
features like an 8MP camera with video streaming capabilities. OpenCv can help us alot in properly utilizing these capabilities for a plethora of amazing applications.
So, if you have just bought your Raspberry bi board especially the Raspberry Pi Zero then all you need is to first set it up.
You may skip this step, if you already have an accessible Rasbperry PI to SSH.

### Fresh installation of Raspberry Pi OS

1. Put the SD card (better > 8GB size) into a USB card reader and plugin to your Computer (any of Mac OS, Windows and Ubuntu/Linux).
2. Go to the official link to download the Raspberry Pi Imager https://www.raspberrypi.org/software/
3. It is really quick and easy to install for Mac OS, Windows, and Ubuntu/Linux. You can watch the 3-steps here: https://www.youtube.com/watch?v=J024soVgEeM
4. Take out and again insert the USB card reader to your PC. This time you will see that the USB device name has changed to Boot.
5. Open the device and find the ```config.txt``` file, open it in a text editor, go to the last line, press enter and add this ```dtoverlay=dwc2``` after the last line, and save it.
6. In the device find the ```cmdline.txt``` and add between two spaces this ```modules-load=dwc2,g_ether```, right after the ```rootwait```, then save and close this .txt as well.
7. Make an empty file in notepad and save it as ```ssh``` and remove the .txt in the name. This fill should exist in the same place as  ```config.txt``` and ```cmdline.txt```.
8. Thats it! now remove the SD card and insert it into the Rasbperry PI SD card slot properly. 
9. Simply plug the USB data cable to your Raspberry Pi card and connect it to the computer. The computer will also provide the power to your card, so no need to insert the power cable.
10. Upon connection the Windows OS will detect a new LAN device, wait for it to be installed and then we need Bonjour.
11. Go to this link, download and install Bonjour https://support.apple.com/kb/dl999?locale=en_GB
12. In the Device Properties of Windows OS, find the USB Serial Port and provide it the drivers from this link USB Ethernet Drivers: https://wiki.moddevices.com/wiki/Troubleshooting_Windows_Connection
13. After that you can view in the device in the Network and Sharing connections. The device is ready to SSH!




