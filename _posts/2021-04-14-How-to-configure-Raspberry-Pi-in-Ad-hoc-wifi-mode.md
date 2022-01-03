---
layout: post
title: How to configure Raspberry Pi in Ad hoc wifi mode
categories: [Raspberry Pi Programming Series]
mathjax: true
featured-img: jeep02
summary:  This tutorial is about configuring RPi in Ad hoc interface
---

Hi friends! Today's tutorial is Part 02 of the Raspberry Pi (RPi) learning series. In this, you will learn how to configure your RPi device in Ad hoc mode or infrastructure-less mode. Yes, that means without any requirement of a wifi router.  That is a significant advantage of Ad hoc mode because it gives us the wifi-router freedom, and a dedicated point-to-point link provides lower latency, which is the best choice for the FPV systems.

<br>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/L0PaW55ZLmw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>

By default, the RPi devices connect to the wifi router upon providing the SSID to them. However, in Ad hoc mode, we have to assign our SSID and the IP address to the device. So, instead of searching and connecting to other wireless networks, the RPi will create its network, which is, of course, ad hoc in nature. We can name this network `RPitest.`

The first thing to do is to SSH to the RPi, just like we did in the previous tutorial. Then use the following to cd to the network directory:

```
cd /etc/network
```
Now, to keep the original interface file for the wifi interface, let's make a backup as:
```
sudo cp interfaces wifi-interface
```
Next is to create a new file for our Ad hoc interface so do the following to make file and edit:

```
sudo nano adhoc-interface
```
Now copy the following and paste in the above file:
```
  auto lo
  iface lo inet loopback
  iface eth0 inet dhcp

  auto wlan0
  iface wlan0 inet static
  address 192.168.1.1
  netmask 255.255.255.0
  wireless-channel 4
  wireless-essid RPitest
  wireless-mode ad-hoc
```
After that, we need to save the file, so press Ctrl+X and then Yes and enter to save it. This configuration of adhoc-interface will let the IP address of our RPi to be `192.168.1.1,` and the netmask will be `255.255.255.0`. You can select more channels as well e.g. 11, but 4 is enough to keep it a lightweight network. The network SSID is `RPitest,` and of course, the mode is `ad-hoc`. You can also add the password using `wireless-key yourpassword`,  and other parameters to this configuration, but you got the idea.

The next step is to let our RPi assign an IP to the device trying to connect to it in Ad hoc mode. Now, please make sure that your RPi is already connected to the wifi to install something in it. For this, we need to run the following command:
```
sudo apt-get install isc-dhcp-server
```
After that, we need to edit this `dhcpd.conf` file using:
```
sudo nano /etc/dhcp/dhcpd.conf
```
Scroll down in this file, and after the end line, copy and paste the following in it:

```
  ddns-update-style interim;
  default-lease-time 600;
  max-lease-time 7200;
  authoritative;
  log-facility local7;
  subnet 192.168.1.0 netmask 255.255.255.0 {
   range 192.168.1.5 192.168.1.150;

  }
```
Again save the file by pressing Ctrl+X and Yes Enter. In the above lines, any new device connected to RPi will be assigned `192.168.1.5` IP address and so on. That's all for the configuration editing part.

The next step is to tell our cute little RPi zero, that after rebooting, which interface should it follow? i.e., wifi-interface or adhoc-interface.

To answer this question, we can always go to the network directory as:
```
cd /etc/network
```
Then, if we want to switch to adhoc-interface use:

```
sudo cp /etc/network/adhoc-interface interfaces
```
Similarly, if we want to switch to wifi-interface use:

```
sudo cp /etc/network/wifi-interface interfaces
```

That's it if you have switched to `adhoc-interface`, then after reboot, you can find an `RPitest` network on your mobile or computer device. Once connected to this network, you can enjoy FPV mode. Also, note that in the adhoc mode your RPi will be on 192.168.1.1 adress. 

Finally, every time you want to update some parameters to the adhoc interface, you must switch to wifi interface and then again switch to adhoc interface to get the results back. 

That's it for this tutorial, enjoy :)

## TIP
We can run any Python code automatically by giving the path to profile, once the RPi is booted.
Open up profile
```
sudo nano /etc/profile
```
Now put the path to `/home/pi/Documents/main.py` the profile page as the last line with python version.

`python3 /home/pi/Documents/main.py &`

Save it and reboot, the code main.py will run automatically once booted.





