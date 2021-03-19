---
layout: post
title: How to easily install OpenCv in Raspberry Pi boards
categories: [Raspberry Pi Programming Series]
mathjax: true
summary: This tutorial is about installing Open CV library in Raspberry Pi 
---



Hi! friends, Raspberry Pi boards come with a lot of powerful
features like an 8MP camera with video streaming capabilities. OpenCv can help us alot in properly utilizing these capabilities for a plethora of amazing applications.
So, if you have just bought your Raspberry Pi board especially the Raspberry Pi Zero then all you need is to set it up for SSH.


You may skip this step, if you already have an accessible Rasbperry Pi to SSH.

### Fresh installation of Raspberry Pi OS

1. Insert the SD card (better > 8GB size) into an USB card reader and plug it to your Computer (any of Mac OS, Windows and Ubuntu/Linux).
2. Go to the official link to download the Raspberry Pi Imager https://www.raspberrypi.org/software/
3. It is really quick and easy to install. You can watch the 3-steps here: https://www.youtube.com/watch?v=J024soVgEeM
4. Take out and again insert the USB card reader to your PC. This time you will see that the USB device name has changed to Boot.
5. Open the device and find the ```config.txt``` file, open it in a text editor, go to the last line, press enter and add this ```dtoverlay=dwc2``` after the last line, and save it.
6. In the device find the ```cmdline.txt``` file and add between two spaces this ```modules-load=dwc2,g_ether```, right after the ```rootwait```, then save and close this .txt as well.
7. Make an empty file in notepad and save it as ```ssh``` and remove the .txt in the name. This fill should exist in the same place as  ```config.txt``` and ```cmdline.txt```.
8. Thats it! now remove the SD card and insert it into the Rasbperry Pi SD card slot properly. 
9. Simply plug the USB data cable to your Raspberry Pi card and connect it to the computer. The computer will also provide the power to your card, so no need to insert the power cable.
10. Upon connection the Windows OS will detect a new LAN device, wait for it to be installed and then we need Bonjour.
11. Go to this link, download and install Bonjour https://support.apple.com/kb/dl999?locale=en_GB
12. In the Device Properties of Windows OS, find the USB Serial Port and provide it the drivers from this link USB Ethernet Drivers: https://wiki.moddevices.com/wiki/Troubleshooting_Windows_Connection
13. After that you can view in the device in the Network and Sharing connections. The device is ready to SSH!


### Acess Raspberry Pi OS using SSH

Our Raspberry Pi board is now configured for the SSH access and its Network device is ready. We can now use two ways to access it:
1. By installing Putty software: https://www.putty.org/
2. By installing Bitvise client software: https://www.bitvise.com/ssh-client-download
3. We recommend Bitvise anyway, but in both cases we require the Host, Port, Username and the Password to access the Pi.
4. By default: Host is ```raspberrypi.local```, Port is ```22```, Username is ```pi```, and Password is ```raspberry```, enter these values in the GUI and hit Login.
5. Once login, you can now visualize ```pi@raspberrypi:~ $``` in the Command or Terminal window, enter ```ls``` to view your files. Now that we are able to access the PI, the next step is to enable its wifi so that we can download the opencv using a pip installer.

Again you may skip this step if you can access the PI using SSH and your wifi is enabled.

### Update the Raspberry Pi OS 

1. After the ```pi@raspberrypi:~ $``` in the Terminal window, we can write ```sudo raspi-config``` and hit enter to open the configuration window as below:

```
│                 1 System Options       Configure system settings                                 │
│                 2 Display Options      Configure display settings                                │
│                 3 Interface Options    Configure connections to peripherals                      │
│                 4 Performance Options  Configure performance settings                            │
│                 5 Localisation Options Configure language and regional settings                  │
│                 6 Advanced Options     Configure advanced settings                               │
│                 8 Update               Update this tool to the latest version                    │
│                 9 About raspi-config   Information about this configuration tool                 |
```
Note that you can go back in this window by pressing ```Esc``` key.
2. Select  ```1 System Options``` using the arrow keys and press Enter
3. Now select the Wireless LAN, and Enter the Wifi SSID and the Password and press Enter
```
|                 S1 Wireless LAN      Enter SSID and passphrase                                   │
│                 S2 Audio             Select audio out through HDMI or 3.5mm jack                 │
│                 S3 Password          Change password for the 'pi' user                           │
│                 S4 Hostname          Set name for this computer on a network                     │
│                 S5 Boot / Auto Login Select boot into desktop or to command line                 │
│                 S6 Network at Boot   Select wait for network connection on boot                  │
│                 S7 Splash Screen     Choose graphical splash screen or text boot                 │
│                 S8 Power LED         Set behaviour of power LED                                  |
```
This will enable the Wifi access of Raspberry Pi board to the internet. 

4. Once the wifi access is enable, after the ```pi@raspberrypi:~ $``` in the Terminal window, we need to update the system by using following commands:

```sudo apt-get update && sudo apt-get upgrade```

5. It may take a while depending on your speed of the internet.

Please note that in the latest Pi operating systems the Python2 and Python3 are already installed.

### Using pip easily install the OpenCV

Although this method is way too simple, yet it still requires some dependencies to install as shown below.

```sudo apt-get install libhdf5-dev libhdf5-serial-dev```

```sudo apt-get install python3-h5py```

```sudo apt-get install libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5```

```sudo apt-get install libatlas-base-dev```

```sudo apt-get install libjasper-dev```
Now simply install the opencv using Python3 as below:

```sudo pip3 install opencv-contrib-python==3.4.4.19```

To check if OpenCV is correctly installed, simply type ``` python3``` in the terminal window and then

``` import cv2```

If no error appears, that means your cv2 is ready to be used. If you have any questions or suggestions, please ask in the comments below. Cheers and have a nice day!










