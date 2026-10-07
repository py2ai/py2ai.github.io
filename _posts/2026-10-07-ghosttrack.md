---
layout: post
title: "GhostTrack: A Minimalist OSINT Console for IP, Phone, and Username Recon - Inside HunxByts/GhostTrack"
description: "A source-level tour of GhostTrack, the single-file Python OSINT console by HunxByts that pairs an ipwho.is geolocation query, a phonenumbers-powered phone report and a two-dozen-site username sweep behind one numbered terminal menu - and why its small surface makes it a good case study in how open-source OSINT tools are engineered."
date: 2026-10-07
header-img: "img/post-bg.jpg"
permalink: /ghosttrack/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/ghosttrack/hunxbyts-ghosttrack-architecture.svg
tags: [OSINT, Python, Security, Recon]
categories: [AI, Open Source]
keywords: GhostTrack, HunxByts, OSINT, IP tracker, phone number lookup, username search, phonenumbers, ipwho.is, Python, information gathering, Termux, open source security tools
author: "PyShine"
---

Open-source intelligence tools rarely come smaller than GhostTrack. Published on GitHub by HunxByts and described in its README as "a useful tool to track location or mobile number, so this tool can be called osint or also information gathering," the whole project is one Python file, a two-line requirements list and a folder of screenshots. Yet inside that single [GhostTR.py](https://github.com/HunxByts/GhostTrack/blob/main/GhostTR.py) file lives a complete console application with four investigation modes: an IP tracker powered by the ipwho.is geolocation API, a phone-number report built on Google's phonenumbers library, a username sweep across two dozen social platforms, and a quick echo of your own public IP. Version 2.2, the current release noted in the README, runs unchanged on desktop Linux and on Termux, Android's terminal emulator.

What makes GhostTrack worth reading is not raw capability - larger OSINT frameworks dwarf it - but its architecture as a teaching artifact. Every technique that the big tools industrialize is visible here in its simplest form: calling a public geolocation endpoint and mapping the JSON fields into a readable report, parsing an E.164 phone number into carrier, region and timezone attributes, and probing profile URLs with plain HTTP GET requests and status-code checks. If you have ever wondered how a phone lookup tool knows a number is a valid mobile line, or how a username search decides a profile exists, GhostTrack answers both in a few hundred readable lines.

A note on intent before the tour: GhostTrack is squarely in the OSINT tradition of tools intended for security research, authorized investigations and learning how information aggregates across public surfaces. The project itself points you toward combining it with other tools for legitimate security workflows, and every capability it has - IP geolocation, phone metadata, public profile checks - draws on data that is already public. This article reads the repository the way an engineer would: what each module does, how the pieces connect, and what the design choices teach. It is not a manual for investigating people, and none of the techniques here should be pointed at anyone without their consent or a lawful basis.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ghosttrack/hunxbyts-ghosttrack-overview-architecture.svg" alt="Architecture overview of GhostTrack showing the main loop, the numbered options registry, the four tracker modules and their external data sources: the ipwho.is API, api.ipify.org, the phonenumbers library and the social profile probe list" style="max-width:100%;">
</div>
<p><em>Architecture overview of GhostTrack: one menu loop dispatching four trackers onto four external data sources.</em></p>

Reading the overview from left to right:

- The **main() loop** in [GhostTR.py](https://github.com/HunxByts/GhostTrack/blob/main/GhostTR.py) paints the menu and the ghost banner, reads a numbered choice, and hands it to the dispatcher; every error path - a non-number, an unknown option, a Ctrl+C - routes back into the same loop.
- The **options registry** is a Python list of dictionaries, each pairing a menu number, a label and the function that implements it, which is the entire routing table of the application.
- The **IP Tracker** option sends a GET request to `http://ipwho.is/{ip}` and prints roughly thirty fields from the response, from country and city down to ASN, ISP and timezone offset, plus a Google Maps link built from the returned latitude and longitude.
- The **Show Your IP** option asks `https://api.ipify.org/` for the caller's own public address and prints it - a one-request sanity check before any deeper work.
- The **Phone Number Tracker** feeds the target number to the phonenumbers library through [requirements.txt](https://github.com/HunxByts/GhostTrack/blob/main/requirements.txt), which returns region, carrier, validity, timezone and several standardized number formats.
- The **Username Tracker** iterates a hard-coded list of twenty-four platform entries, formats each URL template with the target handle, and marks a platform as a hit when the profile page answers HTTP 200.

## Why You Need This

The first reason is legibility. Most OSINT suites bury their logic behind layers of plugins, queues and configuration files, which is exactly right for production use and exactly wrong for learning. GhostTrack inverts that: the IP tracker is one requests.get call followed by a block of print statements that names every field it uses, so you can see precisely what a geolocation API returns - the connection block with ASN, organization, ISP and domain, the timezone block with UTC offset and DST flag, the flag emoji, the calling code. Once you have read it here, reading any commercial IP enrichment dashboard stops being magic.

The second reason is the phone report as a lesson in number metadata. The phonenumbers library, maintained under Google's umbrella as the Python port of libphonenumber, can parse an international number into structured attributes; GhostTrack's phone module asks it for the region code, the carrier name, the geographic description, the time zones, validity and possibility checks, the E.164, international and mobile-dialing formats, and the line type classification between mobile, fixed-line and other. The tool's default parsing region is ID - Indonesia - reflecting the author's context, and the README shows the resulting report. Understanding what a phone number carries and does not carry in its digits is a small but durable piece of security literacy.

The third reason is the design pattern underneath the menu. GhostTrack is a complete, working example of the registry-dispatcher pattern: options live in data, the dispatcher resolves numbers to functions through one guard and one resolver, and a decorator reprints the banner before every tracker runs. The whole user interface is ANSI color codes and a clear() helper that calls cls on Windows and clear elsewhere. For anyone teaching terminal application structure - or porting a console tool to another language - this file is a compact reference, and because it runs on Termux it doubles as an accessible first read for students who only own a phone.

## How It Works

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ghosttrack/hunxbyts-ghosttrack-architecture.svg" alt="Detailed architecture of GhostTrack showing the entry and presentation layer, the menu engine with decorator and dispatcher, the four tracker modules with their report printers, and the external services and libraries they call" style="max-width:100%;">
</div>
<p><em>Detailed architecture of GhostTrack, from the ANSI-painted menu engine down to each library call.</em></p>

### Understanding the Architecture

**Entry and presentation.** The script's entry point guards main() with a KeyboardInterrupt handler so Ctrl+C prints a short exit message instead of a traceback. main() calls clear(), then the option() painter, which writes the ASCII GHOSTTRACK banner and the numbered menu through stderr.writelines - a small detail that keeps the banner on the error stream rather than stdout. All color comes from a block of ANSI escape variables (white, green, red, yellow and friends) that the whole file interpolates into f-strings. run_banner(), the ghost-face art shown before each tracker, is invoked not by the menu but by the @is_option decorator, which wraps every tracker function, clears the screen, sleeps briefly, and then calls the wrapped function.

**The menu engine.** The options list at the middle of the file pairs each menu number with a label and a function reference - option 1 to IP_Track, 2 to showIP, 3 to phoneGW, 4 to TrackLu and 0 to exit. is_in_options() validates a choice before anything runs; call_option() walks the list and invokes the matching function, raising ValueError when nothing matches. execute_option() wraps that resolution in its own error handling: a ValueError prints the message, sleeps two seconds and retries the same choice; a KeyboardInterrupt exits cleanly; success pauses with a "Press enter to continue" prompt and recurses back into main(), which redraws the menu. It is a stateless loop - no classes, no globals beyond the color variables - and that statelessness is why the file stays readable end to end.

**The IP tracker.** IP_Track() reads the target address, requests `http://ipwho.is/{ip}`, and json.loads the response body. The report then prints in three bands: the geographic and political fields (type, country and code, city, continent, region, latitude and longitude, EU membership, postal code, calling code, capital, borders, flag emoji); the connection block (ASN, organization, ISP, domain); and the timezone block (id, abbreviation, DST flag, offset, UTC and current time). Two lines turn the latitude and longitude into a clickable Google Maps URL with an 8z zoom, which is the entire "tracking" mechanism - the API knows where the network is registered, and the tool renders that knowledge.

**The phone tracker.** phoneGW() parses the input with phonenumbers.parse() using ID as the default region, then fans out into the library's submodules: carrier.name_for_number() for the operator, geocoder.description_for_number() for the registered location, timezone.time_zones_for_number() for the zones, is_valid_number() and is_possible_number() for sanity verdicts, and the formatting family for international, mobile-dialing and E.164 renderings of the same number. A type check against PhoneNumberType.MOBILE and FIXED_LINE classifies the line, with anything else reported as "another type of number". The README's screenshot shows the resulting report for a sample number; the architecture is a single parse followed by parallel attribute lookups, all printed in one block.

**The username tracker.** TrackLu() defines a social_media list of twenty-four dictionary entries - twenty-three distinct platforms, with Snapchat appearing twice - spanning Facebook, Twitter, Instagram, LinkedIn, GitHub, Pinterest, Tumblr, YouTube, SoundCloud, Snapchat, TikTok, Behance, Medium, Quora, Flickr, Periscope, Twitch, Dribbble, StumbleUpon, Ello, Product Hunt, Telegram and We Heart It. Each entry carries a URL template with a {} placeholder; the loop formats it with the requested username, issues a plain requests.get, and records the platform as a hit when the status code is 200, or the string "Username not found" otherwise. Results print as a single report block, hits and misses interleaved in list order. This is the simplest possible existence check - no cookies, no rate limiting, no headless browser - which also means platforms that answer 200 for missing accounts can produce false positives, a limitation worth knowing in any username-sweep design.

**Failure handling as a whole.** The tracker bodies differ in robustness: the username tracker wraps its whole loop in a try/except that prints the exception and returns, while the IP and phone trackers let a malformed response surface as a traceback that the outer main() loop then catches. For a tool this size that is an honest trade - the code stays linear and teachable, and the menu loop guarantees you always land back at the prompt.

Follow one session end to end and the layers connect like this: main() paints the menu from the options registry, execute_option() validates and resolves your number, the @is_option decorator clears the screen and reprints the ghost, and the chosen tracker makes its requests - one to ipwho.is or ipify, one parse into phonenumbers, or a run of GETs across the platform list - before pausing and returning control to the same menu loop that started it.

## Advantages

- **True single-file design.** The entire application is one Python module plus a two-dependency requirements list (requests, phonenumbers); there is no package structure to learn, no configuration file, and no build step.
- **Runs where students actually are.** The README gives both apt-based Linux and Termux instructions, so the same tool runs on a laptop and an Android phone without changes.
- **Registry-driven menus.** Adding a fifth tracker means appending one dictionary to the options list and one decorated function - the dispatcher, validator and banner decorator need no edits.
- **Field-complete reporting.** The IP report prints the full ipwho.is payload, including connection and timezone sub-objects that most quick scripts skip, which makes the tool a convenient way to inspect what the API actually returns.
- **Zero configuration.** Nothing to register, no API key to obtain; ipwho.is, ipify and the social probes are all keyless endpoints, and phonenumbers is an offline library.
- **Readable error paths.** Every user-facing error - bad choice, non-number input, interrupted session - is handled in one visible place, which makes the control flow easy to trace.

## Benefits

- **A compact OSINT curriculum.** Geolocation payloads, phone-number metadata and profile existence checks appear here in their rawest form, each in a function short enough to read over coffee.
- **A reusable skeleton.** The registry-dispatcher-decorator trio ports directly to any other console tool; replace the four trackers with your own commands and the file structure survives unchanged.
- **Cross-platform by construction.** The clear() helper and pure-Python dependencies mean Windows, Linux and Termux all behave identically - a small but practical lesson in portable console design.
- **Honest about data sources.** Because every lookup is a visible HTTP call or a local library query, users learn which facts come from a network API and which are computed offline from the number itself.
- **A baseline for building further.** Anyone extending the tool - new platforms, new providers, JSON export - starts from a codebase small enough to modify in one sitting, which is exactly how many larger OSINT projects begin.
- **Educational framing built in.** The README presents the tool as information gathering for learning, and the project links companion tools for authorized security workflows rather than advertising covert capability.

## Usage

Installation follows the README exactly, for Debian-family Linux:

```bash
sudo apt-get install git
sudo apt-get install python3
```

or, on Termux:

```bash
pkg install git
pkg install python3
```

Then clone, install the two dependencies and launch:

```bash
git clone https://github.com/HunxByts/GhostTrack.git
cd GhostTrack
pip3 install -r requirements.txt
python3 GhostTR.py
```

The console presents five numbered choices - IP Tracker, Show Your IP, Phone Number Tracker, Username Tracker and Exit. The IP tracker asks for an address and prints the full geolocation report with a Google Maps link; the README notes that the IP menu can be combined with the companion Seeker tool, which collects a target's IP during an authorized security test and hands it to this menu for enrichment. The phone tracker expects an international number such as +6281xxxxxxxxx and prints the metadata report; the username tracker asks for a handle and lists which of the twenty-four tracked platforms answered 200. Use it on your own addresses, numbers and handles, or within the bounds of an authorized engagement - that is the audience the project writes for.

## Conclusion

GhostTrack is the smallest tool we have toured in this series, and that is precisely its value. In one Python file it demonstrates the three data moves that underpin most of the OSINT category - enrich an address through a geolocation API, decompose a phone number into structured attributes, and probe public profile URLs for existence - wrapped in a menu architecture clean enough to teach from. It will not replace the heavyweight frameworks, and it does not try to; its README, its screenshots and its Termux instructions all aim at accessibility and learning. As a first stop for understanding how these tools actually work - and for seeing how much engineering a handful of HTTP calls and one library can carry - GhostTrack earns its place on the shelf.

Links:

- Repository: [https://github.com/HunxByts/GhostTrack](https://github.com/HunxByts/GhostTrack)
- Main module: [GhostTR.py](https://github.com/HunxByts/GhostTrack/blob/main/GhostTR.py)
- Dependencies: [requirements.txt](https://github.com/HunxByts/GhostTrack/blob/main/requirements.txt)
- Companion tool referenced in the README: [thewhiteh4t/seeker](https://github.com/thewhiteh4t/seeker)
- phonenumbers library: [https://pypi.org/project/phonenumbers/](https://pypi.org/project/phonenumbers/)
