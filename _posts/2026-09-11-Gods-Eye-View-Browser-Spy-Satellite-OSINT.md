---
layout: post
title: "God's Eye View: A Spy Satellite Simulator in Your Browser with Real OSINT Data"
description: "God's Eye View is an open source browser-based intelligence console for planet Earth. It renders a photorealistic 3D globe with live aircraft, ships, satellites, earthquakes, traffic, and public cameras, all driven by real public data sources. Built with vanilla JavaScript, CesiumJS, and Vite (no framework), it starts keyless with Esri satellite imagery and 9 live layers, then powers up with optional Cesium ion, Google Maps, and OpenAI keys for 3D terrain, place search, and hands-free voice control. The voice agent uses the OpenAI Realtime API with 28 tools across 4 jobs (direct, annotate, identify, frame), pulls live scene context before answering, and never lets the API key touch the browser. Cockpit mode rides inside tracked flights with 7 GLSL sensor styles (CRT, NVG, FLIR, Noir, Snow), a detection overlay, and a military HUD. MIT licensed, 24.4k stars, #1 on GitHub Trending August 2026. This post breaks down the architecture, the 13 data layers, the voice pipeline, and the cockpit experience."
date: 2026-09-11
header-img: "img/post-bg.jpg"
permalink: /Gods-Eye-View-Browser-Spy-Satellite-OSINT/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - God's Eye View
  - OSINT
  - CesiumJS
  - 3D Globe
  - Geospatial
  - Open Source
  - Spatial Intelligence
  - Voice Control
author: PyShine
---

## What is God's Eye View

God's Eye View is a real-time intelligence console for planet Earth. It renders a photorealistic 3D globe in your browser with live aircraft, ships, satellites, earthquakes, traffic, and public cameras, all driven by real public data sources. The code is on GitHub at [bilawalsidhu/gods-eye-view](https://github.com/bilawalsidhu/gods-eye-view), MIT licensed, with 24.4k stars and counting.

The project reached #1 on GitHub Trending (daily and weekly) in August 2026, was #8 Product of the Day on Product Hunt (hunted by Chris Messina, creator of the hashtag), and was endorsed by Brendan Eich, creator of JavaScript. The tagline says it all: "A spy-satellite simulator in your browser, then you realize the sources are public and the data is real."

Half the magic is that it looks like a forbidden cockpit. The other half is that every line of code is inspectable. Flight transponders, ship beacons, orbital elements, seismographs, and public cameras already tell us a lot about the world. God's Eye View puts them in the same place, so you can move between a global picture and an individual aircraft, ship, or street. It runs locally in your browser, with source code you can inspect and extend.

## System Architecture

God's Eye View is built with vanilla JavaScript, CesiumJS, and Vite. No framework. The entire app runs locally on localhost, with a server-side proxy that brokers all private API keys.

![God's Eye View system architecture](/assets/img/diagrams/gods-eye-view/gev-architecture.svg)

### Understanding the Architecture

The architecture is deliberately simple and inspectable. The browser layer contains the CesiumJS globe and all rendering logic. There is no React, no Vue, no Angular - just vanilla JavaScript modules organized by responsibility.

**Core Modules**

`main.js` bootstraps the application: it initializes the Cesium viewer, loads Google Photorealistic 3D tiles when available, and registers all data layers. `ui.js` is the runtime UI, handling panels, HUD, visual styles, and the control facade that ties user interactions to the globe. `hud.js` renders the intelligence HUD and the AI scene summary.

`mapStackController.js` handles basemap switching between Google Photorealistic 3D, Esri World Imagery, OSM, and additional Cesium ion-hosted stacks. The keyless default is Esri satellite imagery; OSM is the fallback if Esri is unreachable.

**Data Layer**

The `data/` directory contains one module per layer: flights, vessels, satellites, earthquakes, CCTV, traffic, fires, radio, bikeshare, military installations, and more. Each layer is a separate module, so you can add your own by following the same pattern. The `data/local_data/` folder ships bundled static datasets (datacenters, dams, submarine cables, Natural Earth regions, SF neighborhoods) for an out-of-the-box experience.

**Voice Agent**

The `voice/` directory contains the OpenAI Realtime session and 28 voice tools. The voice agent pulls live scene context (coordinates, street names, active layers, view scale) before answering, and at street level it reads a viewport screenshot to identify legible signage and building names. It is instructed never to hallucinate labels.

**Server-side Proxy**

Every API that touches a private key (OpenAI, AISStream, OpenSky OAuth, camera frames, FIRMS, TomTom) is brokered through a hardened server-side proxy with SSRF protection, response caps, and sanitized errors. The only keys the browser sees are Google Maps and Cesium ion, both of which must be restricted at the provider. The `OPENAI_API_KEY` never touches the browser; the client only gets a short-lived session token.

**Key Design Decisions**

The "no framework" choice is deliberate. Vanilla JavaScript plus CesiumJS plus Vite means the code is fast to read, fast to hack on, and has no abstraction layer between the developer and the globe. A point-in-time M5/Chrome capture measured a median 1.86-second cold start. The app starts keyless: Esri satellite imagery plus 9 live layers work without any API key. Keys are upgrades, not prerequisites.

## The 13 Data Layers

God's Eye View has 13 data layers and map sources. Eleven have a keyless path. Some offer additional capabilities with a provider key.

![God's Eye View data layers](/assets/img/diagrams/gods-eye-view/gev-data-layers.svg)

### Understanding the Data Layers

The data layers are organized by authentication requirement: keyless (green, no API key needed), free key (yellow, free signup required), and metered (red, pay-per-use). This three-tier system is central to the project's design philosophy: the app must be useful the moment you clone it, and every key is an upgrade, not a gate.

**Keyless Layers (11 layers, no API key)**

The Map Stack uses Esri World Imagery as the keyless satellite basemap, with OSM as the automatic fallback. Live Flights pulls 11,000+ live aircraft from OpenSky Network with adsb.lol as a bounded fallback when OpenSky has no usable snapshot. Military Flights shows ADS-B military traffic in amber from adsb.lol. Satellites tracks an 838-object catalog from CelesTrak, color-coded by class, with a DENSE chip that drops in the whole Starlink shell. Earthquakes shows global seismic activity from USGS for the last 24 hours.

Traffic is simulated along real OSM roads using aggregate location data. The CCTV Mesh projects approximately 800 public cameras into the 3D space (Austin, California via Caltrans, London via TfL). Camera positions are published; poses are estimated priors you calibrate by dragging a gizmo on the camera itself. Radio provides 750 geolocated world radio stations with an analog tuner. Bikeshare shows live station availability from GBFS. Mapped Installations shows viewport-bounded military-site context from community OpenStreetMap mapping. Space Missions shows rolling 30-day launches from Launch Library 2.

**Free Key Layers (3 layers, free signup)**

Live Vessels pulls thousands of ships worldwide from AISStream.io (free signup). Active Fires shows NASA FIRMS detections for the trailing 24 hours (free FIRMS MAP_KEY). Traffic Flow adds live congestion colors via TomTom's free tier (200K tile requests per month) - without a key, the traffic layer runs its built-in simulation.

**Metered Layers (2 layers, pay-per-use)**

Google Photorealistic 3D provides the direct Google 3D tiles and in-app place search via the Google Maps Platform (metered, URL-restrict your key). The Voice Agent uses the OpenAI Realtime API for 28 voice tools and the AI HUD summary (metered, key stays server-side).

**Bundled Static Data**

The repo ships bundled static datasets for an out-of-the-box experience: 4,351 datacenters (ODbL 1.0 from OpenStreetMap), 704 dams (ODbL 1.0 from OpenInfraMap/OSM), 712 submarine cables (CC BY-NC-SA 3.0 from TeleGeography - non-commercial, must be removed for commercial use), 1,338 Natural Earth physical regions (public domain), and 41 San Francisco neighborhood polygons (PDDL 1.0 from DataSF). Each bundled dataset has its own provenance README and license carve-out.

**The Basemap Ladder**

The basemap ladder determines what globe you get. With nothing, you get Esri World Imagery satellite basemap and keyless terrain in 2D, with OSM as fallback. With a free Cesium ion token, you get Google Photorealistic 3D cities and world terrain (eligible personal, non-commercial use). With a Google Maps key, you get the same 3D direct from Google plus in-app place search (the billing-enabled, metered route).

## The Voice Agent: 28 Tools, 4 Jobs

The voice agent is where God's Eye View crosses from a visualization tool into an interactive intelligence console. It uses the OpenAI Realtime API, and the same key drives the AI HUD summary: a terse, five-word intelligence-style readout of the current view that regenerates as you move.

![God's Eye View voice pipeline](/assets/img/diagrams/gods-eye-view/gev-voice-pipeline.svg)

### Understanding the Voice Pipeline

The voice pipeline is built around a security-first design. The `OPENAI_API_KEY` never touches the browser. The server-side proxy brokers the key, and the client only receives a short-lived session token. The agent only confirms actions that succeeded - it does not claim a command executed if it failed.

**Scene Context First**

Before answering any question, the agent pulls live scene context: coordinates, street names, active layers, and view scale. At street level, it reads a viewport screenshot to identify legible signage and building names (visual grounding). It is instructed never to hallucinate labels. This means you can ask "what city is this?" mid-flight and it knows.

**The 28 Tools Across 4 Jobs**

The 28 voice tools are organized into four jobs, and the commands come straight from the product's voice test suite and tool playbook.

Job 1: Direct It - drone-operator camera verbs. "Take me to Tokyo." "Orbit around this area slowly." "Draw the walking route from the Capitol to Zilker Park" then "Fly the route we just drew." "Zoom out to a globe view."

Job 2: Annotate It - a whiteboard over the real world. "Outline the state of Texas" draws the actual enclosing boundary polygon, not a circle. "Annotate the Texas State Capitol and its grounds" draws the real boundary. "How far is the Eiffel Tower from the Louvre?" - a connector arrow appears and it speaks the distance. Everything persists until you say "clear the map."

Job 3: Identify It - entity Q&A. Click any plane, ship, or datacenter and ask "what's this?" It answers using the object's live telemetry. The agent knows what it is looking at because it pulled scene context first.

Job 4: Frame It - cinematic framing. "Show me the planes overhead" pulls the camera back, angles it, and frames the live traffic like a director.

**Security Model**

The security model is explicit. The `OPENAI_API_KEY` never touches the browser. The client gets a short-lived session token only. The agent only confirms actions that succeeded. Voice is optional - the entire app runs without an OpenAI key; the mic button just reports voice is unavailable. The server binds to localhost, and Provider Settings answers requests only from your machine.

## Cockpit Mode and Sensor Styles

Cockpit mode is the experience that made the project go viral. You click-track a live airliner, hop into the cockpit, and ride it down with real terrain holding underneath, all the way to the runway. Sensor styles come along for the ride, and Contacts keeps the 250 km roster one click away: jump plane to plane and fall straight into the next cockpit.

![God's Eye View cockpit and sensors](/assets/img/diagrams/gods-eye-view/gev-cockpit-sensors.svg)

### Understanding the Cockpit

The cockpit carries its own briefing strip: nearby live signals, regional headlines (Google News RSS with GDELT fallback), and real local weather from Open-Meteo. An opt-in WX mode renders volumetric clouds from actual observations around your aircraft.

**7 Sensor Styles (keys 1-7)**

GLSL shaders render the whole live planet through different sensors. Key 1 is the normal globe (standard satellite imagery). Key 2 is CRT (retro green phosphor look). Key 3 is NVG (night vision green). Key 4 is FLIR/thermal (heat-signature view). Key 5 is Noir (black and white film). Key 6 is Snow (winter overlay). Key 7 is Ironbow FLIR (advanced thermal palette). Switch mid-flight: NVG into Ironbow FLIR while riding a tracked plane.

**3D Aircraft Models**

The 3D hangar ships real per-class aircraft models: 787, ATR-72, Citation, Bell 206, MQ-9. A tracked contact swaps from a glyph to a 3D model as you close in. At airport level, grounded contacts show as 3D aircraft on the taxiways with taxi trails.

**Detection Overlay and Military HUD**

The detection overlay (key D) draws screen-space bounding boxes and IDs on everything in view. The military HUD (key H) is a tactical heads-up display with intelligence-style telemetry. Global Context stages the full situational picture with one switch, and you get your exact view back when you leave.

**How the Globe Handles Live Data**

Aircraft and ships point along their true real-world heading at every camera angle via per-frame screen-space course projection. No spinning, no viewport-locking. Live feeds arrive every 15-30 seconds; the globe renders one interval behind real time and interpolates between known fixes. Dead reckoning fills the gaps. Satellites use SGP4 propagation with orbit rings that stay locked to their satellites via GMST realignment - no drift, no per-second flicker. Entity heights run through a real vertical datum - geoid-aware, sampled against the rendered terrain mesh - so aircraft park on aprons and cameras stand on street corners instead of floating.

**Share Links**

Camera, style, layers, and even one tracked target serialize into a URL. A live target is a handoff, not a bookmark. When someone opens your share link, the globe reorients to the same camera position, same sensor style, same active layers, and starts tracking the same live entity (if it is still in range).

## Quick Start

Two paths. Both open the same app with Esri satellite imagery and keyless terrain.

**Path 1: One click, no terminal (Pinokio)**

Install or update [Pinokio](https://pinokio.co/apps/github-com-bilawalsidhu-gods-eye-view) to 8.2 or later. Click Install, then Start. Available on Windows, macOS, and Linux.

**Path 2: Terminal**

Use Node.js 24.x (24.14.0 or later) or 26.x. The setup doctor warns about Node 25, which is end-of-life.

```bash
git clone https://github.com/bilawalsidhu/gods-eye-view.git
cd gods-eye-view
npm ci
npm run doctor
npm run dev
```

Open `http://localhost:4173`. Choose Live Contacts, Space Missions, Environmental, or Explore Manually from the first-run panel.

Then power it up: click the POWER UP chip in the bottom-right corner. Provider Settings lists every supported key, what it switches on, and where to get it. Paste, hit SAVE KEYS, and the app restarts itself with the new capability on. Once everything is configured, the chip reads POWERED UP.

## Key Features

| Feature | Description |
|---------|-------------|
| Photorealistic 3D globe | CesiumJS + Google Photorealistic 3D Tiles or Esri satellite imagery |
| 13 data layers | 11 keyless: flights, military, satellites, earthquakes, traffic, CCTV, radio, bikeshare, installations, space missions, map stack |
| Live aircraft | 11,000+ from OpenSky + adsb.lol, interpolated between fixes |
| Live vessels | Thousands of ships from AISStream.io (AIS) |
| Live satellites | 838 from CelesTrak, SGP4 propagation, Starlink shell |
| CCTV mesh | ~800 public cameras projected into 3D space, calibrated by gizmo |
| Cockpit mode | Ride inside tracked flights, terrain holds all the way down |
| 7 sensor styles | CRT, NVG, FLIR/thermal, Noir, Snow, Ironbow via GLSL |
| 3D aircraft models | 787, ATR-72, Citation, Bell 206, MQ-9 |
| Detection overlay | Screen-space bounding boxes and IDs on everything in view |
| Military HUD | Tactical heads-up display with intelligence-style telemetry |
| Voice agent | OpenAI Realtime API, 28 tools, 4 jobs, scene context first |
| Voice whiteboard | Speak boundary polygons, marks, and routes onto the world |
| AI HUD summary | 5-word intelligence readout, regenerates as you move |
| Share links | Camera, style, layers, tracked target serialize into URL |
| No framework | Vanilla JS + CesiumJS + Vite, 1.86s cold start |
| Keyless start | Esri + OSM + 9 live layers work with no API key |
| Server-side keys | All private keys brokered through hardened proxy |
| MIT licensed | 24.4k stars, #1 GitHub Trending August 2026 |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No 3D cities | No Cesium ion token or Google Maps key | Add a free Cesium ion token (eligible personal non-commercial) or a Google Maps key |
| Voice unavailable | No OpenAI key | The entire app runs without it; add an OpenAI key for voice + AI HUD |
| Esri basemap not loading | Esri service unreachable | OSM takes over automatically as fallback |
| Camera floats above ground | Terrain unavailable | App falls back to flat EllipsoidTerrainProvider; Re:Earth Terrain (Mapterhorn) is the keyless terrain source |
| Pinokio install fails | Pinokio version below 8.2 | Update Pinokio to 8.2 or later; the launcher installer was fixed in 8.2 |
| Node version wrong | Node 25 is end-of-life | Use Node.js 24.x (24.14.0+) or 26.x; run npm run doctor |
| Submarine cables in commercial use | CC BY-NC-SA license | Delete src/data/local_data/telegeography_submarine_cables/ or license from TeleGeography |
| No vessels on globe | No AISStream key | Sign up free at aisstream.io; add key via POWER UP panel |
| No active fires | No FIRMS MAP_KEY | Get a free key at firms.modaps.eosdis.nasa.gov |
| Traffic not live | No TomTom key | Without key, traffic runs built-in simulation on OSM roads |

## Conclusion

God's Eye View is what happens when a developer who cares about open source, public data, and a cinematic experience builds a tool that respects all three. The project's position is earned: #1 on GitHub Trending, 24.4k stars, MIT licensed, and every data source is documented with its license and attribution. The architecture is deliberately simple (vanilla JavaScript, no framework, 1.86-second cold start) so that the code is inspectable and hackable. The security model is deliberate: all private keys stay server-side, the browser only sees keys that can be restricted at the provider, and the voice agent only confirms actions that succeeded.

The 13 data layers cover the major live signals already publicly available about the world: flight transponders, ship beacons, orbital elements, seismographs, public cameras, traffic, fires, radio, and space missions. Eleven of those layers work without any API key. The voice agent's 28 tools across 4 jobs (direct, annotate, identify, frame) turn the globe from a visualization into an interactive intelligence console. Cockpit mode with 7 GLSL sensor styles is the experience that made it viral, but the underlying engineering (per-frame screen-space heading projection, interpolation between fixes, SGP4 with GMST realignment, geoid-aware heights) is what makes it hold up.

For OSINT practitioners, open hardware developers, spatial intelligence researchers, and anyone who wants to see what public data looks like when it is assembled into a single explorable globe, God's Eye View is the reference implementation. The code is at [github.com/bilawalsidhu/gods-eye-view](https://github.com/bilawalsidhu/gods-eye-view).

## Links

- [God's Eye View on GitHub (bilawalsidhu/gods-eye-view)](https://github.com/bilawalsidhu/gods-eye-view)
- [God's Eye View YouTube series](https://www.youtube.com/@bilawalsidhu)
- [God's Eye View on Pinokio](https://pinokio.co/apps/github-com-bilawalsidhu-gods-eye-view)
- [Cesium ion (free token for 3D terrain)](https://cesium.com/ion)
- [CesiumJS (3D globe engine)](https://cesium.com/cesiumjs/)
- [OpenSky Network (live flights)](https://opensky-network.org/)
- [adsb.lol (flight data)](https://adsb.lol/)
- [AISStream.io (live vessels)](https://aisstream.io/)
- [CelesTrak (satellite TLEs)](https://celestrak.org/)
- [USGS Earthquake Hazards Program](https://www.usgs.gov/programs/earthquake-hazards)
- [NASA FIRMS (active fires)](https://firms.modaps.eosdis.nasa.gov/)
- [Radio Browser (world radio)](https://www.radio-browser.info/)
- [Launch Library 2 / The Space Devs](https://thespacedevs.com/)
- [Open-Meteo (weather data)](https://open-meteo.com/)
- [Pinokio (one-click installer)](https://pinokio.co/)
- [Esri World Imagery (keyless basemap)](https://www.esri.com/arcgis-blog/products/arcgis-online/imagery/world-imagery-updated-and-expanded/)
