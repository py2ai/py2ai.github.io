/**
 * Sunshine Intensity Map Application
 * Uses Leaflet with OpenStreetMap for 2D map visualization
 * Shows sun position and solar irradiance based on time and location
 */

// Global state
let map = null;
let currentDate = new Date();
let currentMinutes = 720; // 12:00 noon
let sunIntensityLayer = null;
let sunPathLayer = null;
let panelGridLayer = null;
let marker = null;
let selectedLocation = null; // Store clicked location

// Initialize the application
function initApp() {
    initMap();
    initControls();
    updateSolarData();
    
    // Update time display every second
    setInterval(updateCurrentTimeDisplay, 1000);
}

// Initialize Leaflet map with OpenStreetMap
function initMap() {
    // Create map centered on a default location
    map = L.map('map', {
        center: [37.7749, -122.4194], // San Francisco
        zoom: 12,
        zoomControl: true
    });

    const tileProviders = [
        {
            name: 'OpenStreetMap HOT',
            url: 'https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, Tiles style by <a href="https://www.hotosm.org/" target="_blank">Humanitarian OpenStreetMap Team</a>'
        },
        {
            name: 'OpenStreetMap',
            url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        },
        {
            name: 'CartoDB Light',
            url: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        },
        {
            name: 'Esri Satellite',
            url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
        }
    ];

    let currentProviderIndex = 0;
    let currentLayer = null;
    let fallbackTimeout = null;
    let tileErrorCount = 0;
    const MAX_TILE_ERRORS = 5;

    function createTileLayer(providerIndex) {
        const provider = tileProviders[providerIndex];
        const layer = L.tileLayer(provider.url, {
            maxZoom: 19,
            attribution: provider.attribution
        });
        
        layer.on('tileerror', function(error) {
            tileErrorCount++;
            
            if (tileErrorCount >= MAX_TILE_ERRORS && providerIndex < tileProviders.length - 1) {
                switchToFallback(providerIndex + 1);
            }
        });
        
        layer.on('tileload', function() {
            tileErrorCount = 0;
            const loadingStatus = document.getElementById('loading-status');
            if (loadingStatus) {
                loadingStatus.style.display = 'none';
            }
        });
        
        return layer;
    }

    function switchToFallback(newIndex) {
        if (newIndex >= tileProviders.length) {
            return;
        }
        
        if (currentLayer) {
            map.removeLayer(currentLayer);
        }
        
        currentProviderIndex = newIndex;
        tileErrorCount = 0;
        currentLayer = createTileLayer(newIndex);
        currentLayer.addTo(map);
    }

    currentLayer = createTileLayer(0);
    currentLayer.addTo(map);

    const osmHotLayer = currentLayer;
    const osmLayer = createTileLayer(1);
    const cartoDBLayer = createTileLayer(2);
    const cartoDBDarkLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
    });
    const esriSatelliteLayer = createTileLayer(3);
    const esriTopoLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 19,
        attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community'
    });

    // Layer control
    const baseLayers = {
        'OpenStreetMap HOT': osmHotLayer,
        'OpenStreetMap': osmLayer,
        'CartoDB Light': cartoDBLayer,
        'CartoDB Dark': cartoDBDarkLayer,
        'Esri Satellite': esriSatelliteLayer,
        'Esri Topo': esriTopoLayer
    };

    L.control.layers(baseLayers, {}, { position: 'topright' }).addTo(map);
    
    // Create sun intensity layer separately (not in layer control)
    sunIntensityLayer = L.layerGroup().addTo(map);
    
    // Initialize sun path overlay (checked by default)
    if (document.getElementById('sun-path-overlay').checked) {
        drawSunPath();
    }

    // Add scale control
    L.control.scale({ position: 'bottomleft' }).addTo(map);

    // Add legend
    addLegend();

    // Map click event
    map.on('click', onMapClick);

    // Try to get user location
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                map.setView([lat, lon], 13);
                updateSolarData();
            },
            (error) => {
            }
        );
    }
}

// Add legend to map
function addLegend() {
    const legend = L.control({ position: 'bottomright' });
    
    legend.onAdd = function(map) {
        const div = L.DomUtil.create('div', 'sun-intensity-legend');
        div.innerHTML = `
            <h4><i class="fas fa-sun"></i> Sun Intensity</h4>
            <div class="color-scale">
                <div style="background: #1a1a2e;" title="0 W/m²"></div>
                <div style="background: #2c3e50;" title="200 W/m²"></div>
                <div style="background: #34495e;" title="400 W/m²"></div>
                <div style="background: #f39c12;" title="600 W/m²"></div>
                <div style="background: #f1c40f;" title="800 W/m²"></div>
                <div style="background: #f9e79f;" title="1000 W/m²"></div>
            </div>
            <div class="labels">
                <span>0</span>
                <span>200</span>
                <span>400</span>
                <span>600</span>
                <span>800</span>
                <span>1000 W/m²</span>
            </div>
        `;
        return div;
    };
    
    legend.addTo(map);
}

// Handle map click
function onMapClick(e) {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;
    
    // Check if precise selection mode is enabled
    const preciseMode = document.getElementById('precise-location-mode');
    if (!preciseMode || !preciseMode.checked) {
        return; // Don't do anything if checkbox is not checked
    }
    
    // Store selected location
    selectedLocation = { lat, lon };
    
    // Update tilt slider to optimal value (minimum 1 degree)
    const optimalTilt = Math.max(1, Math.abs(lat).toFixed(0));
    document.getElementById('tilt-slider').value = optimalTilt;
    document.getElementById('tilt-value').textContent = optimalTilt;
    
    // Remove existing marker
    if (marker) {
        map.removeLayer(marker);
    }
    
    // Add new marker with popup
    marker = L.marker([lat, lon]).addTo(map);
    
    const showTooltips = document.getElementById('show-tooltips');
    if (!showTooltips || showTooltips.checked) {
        marker.bindPopup(`<b>Loading...</b>`).openPopup();
    }
    
    // Show info box
    document.getElementById('location-info').style.display = 'block';
    
    // Update all solar data (this will draw the sun line and update popup)
    updateSolarData();
}

// Initialize controls
function initControls() {
    // Date picker
    const datePicker = document.getElementById('date-picker');
    datePicker.value = currentDate.toISOString().split('T')[0];
    datePicker.addEventListener('change', (e) => {
        currentDate = new Date(e.target.value + 'T12:00:00');
        updateSolarData();
    });

    // Clock slider
    initClockSlider();

    // Search
    initSearch();
    
    // Geolocation
    initGeolocation();
    
    // Solar panel controls
    initPanelControls();
    updateSystemSummary();

    // Quick action buttons
    document.getElementById('btn-noon').addEventListener('click', () => setTime(12 * 60));
    document.getElementById('btn-sunrise').addEventListener('click', setToSunrise);
    document.getElementById('btn-sunset').addEventListener('click', setToSunset);
    document.getElementById('btn-now').addEventListener('click', setToCurrentTime);

    // Precise location mode toggle
    document.getElementById('precise-location-mode').addEventListener('change', (e) => {
        const mapContainer = document.getElementById('map');
        if (e.target.checked) {
            // Change cursor to crosshair when enabled
            mapContainer.style.cursor = 'crosshair';
        } else {
            // Clear selected location and marker when disabled
            selectedLocation = null;
            if (marker) {
                map.removeLayer(marker);
                marker = null;
            }
            // Clear sun line
            if (sunIntensityLayer) {
                sunIntensityLayer.clearLayers();
            }
            // Hide info box
            document.getElementById('location-info').style.display = 'none';
            // Reset cursor
            mapContainer.style.cursor = '';
        }
    });

    document.getElementById('sun-path-overlay').addEventListener('change', (e) => {
        if (e.target.checked) {
            drawSunPath();
        } else if (sunPathLayer) {
            map.removeLayer(sunPathLayer);
            sunPathLayer = null;
        }
    });
    
    // Close info box button
    document.getElementById('close-info-box').addEventListener('click', () => {
        document.getElementById('location-info').style.display = 'none';
        // Clear selection
        selectedLocation = null;
        if (marker) {
            map.removeLayer(marker);
            marker = null;
        }
        if (sunIntensityLayer) {
            sunIntensityLayer.clearLayers();
        }
        document.getElementById('map').style.cursor = '';
    });
}

// Initialize clock slider
function initClockSlider() {
    const clockSlider = document.getElementById('clock-slider');
    const clockHand = document.getElementById('clock-hand');
    const clockAmPm = document.getElementById('clock-am-pm');
    const timeDisplay = document.getElementById('time-display');
    let isDragging = false;

    function updateClockHand(minutes) {
        const degrees = (minutes / 720) * 360;
        clockHand.style.transform = `translateX(-50%) translateY(-100%) rotate(${degrees}deg)`;
        const hours = Math.floor(minutes / 60);
        if (clockAmPm) {
            clockAmPm.textContent = hours < 12 ? 'AM' : 'PM';
        }
    }

    function handleClockInteraction(e) {
        const rect = clockSlider.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        
        let mouseX, mouseY;
        if (e.touches) {
            mouseX = e.touches[0].clientX;
            mouseY = e.touches[0].clientY;
        } else {
            mouseX = e.clientX;
            mouseY = e.clientY;
        }
        
        const dx = mouseX - centerX;
        const dy = mouseY - centerY;
        let angleDeg = Math.atan2(dx, -dy) * (180 / Math.PI);
        if (angleDeg < 0) angleDeg += 360;
        
        let minutes = Math.round((angleDeg / 360) * 720);
        if (minutes >= 720) minutes = minutes % 720;
        
        const currentHours = Math.floor(currentMinutes / 60);
        const newHours = Math.floor(minutes / 60);
        
        if (Math.abs(newHours - (currentHours % 12)) > 6) {
            if (currentHours < 12) {
                minutes += 720;
            }
        } else {
            if (currentHours >= 12 && minutes < 720) {
                minutes += 720;
            }
        }
        
        minutes = Math.max(0, Math.min(1439, minutes));
        currentMinutes = minutes;
        
        const hours = Math.floor(minutes / 60);
        const mins = minutes % 60;
        timeDisplay.textContent = `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
        updateClockHand(minutes);
        updateSolarData();
    }

    clockSlider.addEventListener('mousedown', (e) => {
        isDragging = true;
        handleClockInteraction(e);
    });

    document.addEventListener('mousemove', (e) => {
        if (isDragging) handleClockInteraction(e);
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
    });

    clockSlider.addEventListener('touchstart', (e) => {
        isDragging = true;
        handleClockInteraction(e);
        e.preventDefault();
    });

    document.addEventListener('touchmove', (e) => {
        if (isDragging) handleClockInteraction(e);
    });

    document.addEventListener('touchend', () => {
        isDragging = false;
    });

    updateClockHand(currentMinutes);
}

// Initialize search
function initSearch() {
    const searchInput = document.getElementById('location-search');
    const searchBtn = document.getElementById('search-btn');
    const resultsContainer = document.getElementById('location-results');

    if (!searchInput || !searchBtn || !resultsContainer) {
        return;
    }

    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query) return;

        resultsContainer.innerHTML = '<div class="small text-muted">Searching...</div>';

        const coordMatch = query.match(/^(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)$/);
        if (coordMatch) {
            const lat = parseFloat(coordMatch[1]);
            const lon = parseFloat(coordMatch[2]);
            selectedLocation = { lat, lon };
            map.setView([lat, lon], 14);
            resultsContainer.innerHTML = '';
            updateSolarData();
            return;
        }

        let results = null;

        // Try Photon API first (better CORS support)
        try {
            const photonResponse = await fetch(
                `https://photon.komoot.io/api/?q=${encodeURIComponent(query)}&limit=5`
            );
            if (photonResponse.ok) {
                const photonData = await photonResponse.json();
                if (photonData.features && photonData.features.length > 0) {
                    results = photonData.features.map(f => ({
                        lat: f.geometry.coordinates[1],
                        lon: f.geometry.coordinates[0],
                        display_name: f.properties.name + 
                            (f.properties.city ? ', ' + f.properties.city : '') + 
                            (f.properties.state ? ', ' + f.properties.state : '') +
                            (f.properties.country ? ', ' + f.properties.country : '')
                    }));
                }
            }
        } catch (e) {
        }

        // Fallback to Nominatim if Photon fails
        if (!results || results.length === 0) {
            try {
                const response = await fetch(
                    `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5`
                );
                if (response.ok) {
                    results = await response.json();
                }
            } catch (e) {
            }
        }
        
        if (!results || results.length === 0) {
            resultsContainer.innerHTML = '<div class="alert alert-warning py-1 small">No results found</div>';
            return;
        }

        resultsContainer.innerHTML = results.map(r => `
            <div class="list-group-item list-group-item-action py-1 small" 
                 data-lat="${r.lat}" data-lon="${r.lon}" style="cursor: pointer;">
                ${r.display_name.substring(0, 50)}...
            </div>
        `).join('');

        resultsContainer.querySelectorAll('.list-group-item').forEach(item => {
            const handleSelect = (e) => {
                e.preventDefault();
                e.stopPropagation();
                const lat = parseFloat(item.dataset.lat);
                const lon = parseFloat(item.dataset.lon);
                selectedLocation = { lat, lon };
                map.setView([lat, lon], 14);
                resultsContainer.innerHTML = '';
                updateSolarData();
            };
            item.addEventListener('click', handleSelect);
            item.addEventListener('touchend', handleSelect);
        });
    }

    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });
}

// Initialize geolocation button
function initGeolocation() {
    const getLocationBtn = document.getElementById('get-location-btn');
    const searchInput = document.getElementById('location-search');
    
    if (!getLocationBtn) return;
    
    getLocationBtn.addEventListener('click', function() {
        if (!navigator.geolocation) {
            alert('Geolocation is not supported by your browser');
            return;
        }
        
        getLocationBtn.disabled = true;
        getLocationBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        
        navigator.geolocation.getCurrentPosition(
            function(position) {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                
                map.setView([lat, lon], 14);
                
                selectedLocation = { lat, lon };
                
                // Update tilt slider to optimal value (minimum 1 degree)
                const optimalTilt = Math.max(1, Math.abs(lat).toFixed(0));
                document.getElementById('tilt-slider').value = optimalTilt;
                document.getElementById('tilt-value').textContent = optimalTilt;
                
                if (marker) {
                    map.removeLayer(marker);
                }
                marker = L.marker([lat, lon]).addTo(map);
                
                const showTooltipsGeo = document.getElementById('show-tooltips');
                if (!showTooltipsGeo || showTooltipsGeo.checked) {
                    marker.bindPopup(`<b>Your Location</b><br>Lat: ${lat.toFixed(4)}<br>Lon: ${lon.toFixed(4)}`).openPopup();
                }
                
                searchInput.value = `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
                
                getLocationBtn.disabled = false;
                getLocationBtn.innerHTML = '<i class="fas fa-location-crosshairs"></i>';
                
                updateSolarData();
            },
            function(error) {
                let errorMsg = 'Unable to get your location';
                switch(error.code) {
                    case error.PERMISSION_DENIED:
                        errorMsg = 'Location permission denied. Please enable location access.';
                        break;
                    case error.POSITION_UNAVAILABLE:
                        errorMsg = 'Location information unavailable.';
                        break;
                    case error.TIMEOUT:
                        errorMsg = 'Location request timed out.';
                        break;
                }
                alert(errorMsg);
                getLocationBtn.disabled = false;
                getLocationBtn.innerHTML = '<i class="fas fa-location-crosshairs"></i>';
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 0
            }
        );
    });
}

// Set time helper
function setTime(minutes) {
    currentMinutes = minutes;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    document.getElementById('time-display').textContent = `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
    
    // Update clock hand
    const clockHand = document.getElementById('clock-hand');
    const degrees = (minutes / 720) * 360;
    clockHand.style.transform = `translateX(-50%) translateY(-100%) rotate(${degrees}deg)`;
    
    const clockAmPm = document.getElementById('clock-am-pm');
    if (clockAmPm) {
        clockAmPm.textContent = hours < 12 ? 'AM' : 'PM';
    }
    
    updateSolarData();
}

// Get timezone for coordinates using free API
async function getTimezoneOffset(lat, lon, dateStr) {
    try {
        const timestamp = Math.floor(new Date(dateStr).getTime() / 1000);
        const response = await fetch(`https://maps.googleapis.com/maps/api/timezone/json?location=${lat},${lon}&timestamp=${timestamp}&key=free`);
        return null;
    } catch (e) {
        return null;
    }
}

// Get timezone offset for a location
async function getLocationTimezoneOffset(lat, lon, date) {
    // Use a simple but accurate timezone calculation based on longitude
    // with known timezone exceptions for specific regions
    
    // Iran (UTC+3:30) - specific to Iran only
    if (lon >= 44 && lon <= 64 && lat >= 25 && lat <= 40) {
        // Exclude UAE and other Gulf states (they use UTC+4)
        if (lon >= 51 && lon <= 57 && lat >= 22 && lat <= 26) {
            return 4 * 60; // UAE uses UTC+4
        }
        // Exclude Turkmenistan (uses UTC+5)
        if (lat >= 35 && lon >= 52 && lon <= 67) {
            return 5 * 60; // Turkmenistan uses UTC+5
        }
        return 3.5 * 60; // UTC+3:30, no DST
    }
    
    // Pakistan (UTC+5)
    if (lon >= 60 && lon <= 80 && lat >= 23 && lat <= 37) {
        return 5 * 60; // UTC+5, no DST
    }
    
    // India (UTC+5:30)
    if (lon >= 68 && lon <= 97 && lat >= 6 && lat <= 35) {
        return 5.5 * 60; // UTC+5:30, no DST
    }
    
    // China (UTC+8) - single timezone for entire country
    if (lon >= 73 && lon <= 135 && lat >= 18 && lat <= 53) {
        // Kyrgyzstan exception (UTC+6)
        if (lon >= 73 && lon <= 80 && lat >= 39 && lat <= 44) {
            return 6 * 60;
        }
        return 8 * 60; // UTC+8, no DST
    }
    
    // Japan (UTC+9)
    if (lon >= 122 && lon <= 154 && lat >= 24 && lat <= 46) {
        return 9 * 60; // UTC+9, no DST
    }
    
    // Australia - multiple timezones
    if (lat < -10 && lon > 110 && lon < 155) {
        // Eastern Australia (Sydney, Melbourne, Brisbane)
        if (lon > 140) {
            const month = date.getUTCMonth();
            // DST: Oct-Apr in southern hemisphere
            const isDST = month < 4 || month > 9;
            return (isDST ? 11 : 10) * 60;
        }
        // Central Australia
        if (lon > 129 && lon <= 140) {
            const month = date.getUTCMonth();
            const isDST = month < 4 || month > 9;
            return (isDST ? 10.5 : 9.5) * 60;
        }
        // Western Australia
        return 8 * 60;
    }
    
    // Canada timezones (more accurate than longitude-based)
    if (lat >= 41 && lat <= 84 && lon >= -141 && lon <= -52) {
        const month = date.getUTCMonth();
        const day = date.getUTCDate();
        const isDST = (month > 2 && month < 10) || (month === 2 && day >= 8) || (month === 10 && day < 7);
        // Pacific: British Columbia (west)
        if (lon >= -141 && lon < -125) return (isDST ? -7 : -8) * 60;
        // Mountain: Alberta, BC interior, Yukon, Northwest Territories
        if (lon >= -125 && lon < -102) return (isDST ? -6 : -7) * 60;
        // Central: Saskatchewan, Manitoba, Nunavut (west)
        if (lon >= -102 && lon < -85) return (isDST ? -5 : -6) * 60;
        // Eastern: Ontario, Quebec, Nunavut (east)
        if (lon >= -85 && lon < -66) return (isDST ? -4 : -5) * 60;
        // Atlantic: Maritime provinces
        return (isDST ? -3 : -4) * 60;
    }
    
    // USA timezones
    if (lat >= 24 && lat <= 50 && lon >= -125 && lon <= -66) {
        const month = date.getUTCMonth();
        const day = date.getUTCDate();
        // DST: Mar 8 - Nov 1 (rough)
        const isDST = (month > 2 && month < 10) || (month === 2 && day >= 8) || (month === 10 && day < 7);
        
        if (lon >= -125 && lon < -110) return (isDST ? -7 : -8) * 60; // Pacific
        if (lon >= -110 && lon < -100) return (isDST ? -6 : -7) * 60; // Mountain
        if (lon >= -100 && lon < -85) return (isDST ? -5 : -6) * 60;  // Central
        return (isDST ? -4 : -5) * 60; // Eastern
    }
    
    // Europe (rough approximation)
    if (lat >= 35 && lat <= 70 && lon >= -10 && lon <= 40) {
        const month = date.getUTCMonth();
        const isDST = month > 2 && month < 10;
        const baseOffset = Math.round(lon / 15) * 60;
        return baseOffset + (isDST ? 60 : 0);
    }
    
    // Default: Use longitude-based timezone with DST for northern/southern regions
    const lonOffset = Math.round(lon / 15);
    const baseOffset = lonOffset * 60;
    
    const month = date.getUTCMonth();
    const day = date.getUTCDate();
    
    let isDST = false;
    if (lat > 23.5) {
        // Northern hemisphere (above Tropic of Cancer)
        isDST = (month > 2 && month < 10) || (month === 2 && day >= 8) || (month === 10 && day < 7);
    } else if (lat < -23.5) {
        // Southern hemisphere (below Tropic of Capricorn)
        isDST = month < 4 || month > 9;
    }
    // Tropical regions (between Tropics) typically don't observe DST
    
    return baseOffset + (isDST ? 60 : 0);
}

// Set to sunrise using sunrise-sunset.org API with timezone
async function setToSunrise() {
    let targetLat, targetLon;
    if (selectedLocation) {
        targetLat = selectedLocation.lat;
        targetLon = selectedLocation.lon;
    } else {
        const center = map.getCenter();
        targetLat = center.lat;
        targetLon = center.lng;
    }
    
    const dateStr = document.getElementById('date-picker').value;
    
    try {
        const response = await fetch(`https://api.sunrise-sunset.org/json?lat=${targetLat}&lng=${targetLon}&date=${dateStr}&formatted=0`);
        const data = await response.json();
        
        if (data.status === 'OK') {
            const sunriseUTC = new Date(data.results.sunrise);
            
            // Get local time offset
            const localOffsetMinutes = await getLocationTimezoneOffset(targetLat, targetLon, sunriseUTC);
            
            // Convert UTC to local clock time
            const localHours = sunriseUTC.getUTCHours() + sunriseUTC.getUTCMinutes() / 60 + localOffsetMinutes / 60;
            
            let normalizedHours = localHours;
            while (normalizedHours < 0) normalizedHours += 24;
            while (normalizedHours >= 24) normalizedHours -= 24;
            
            const hours = Math.floor(normalizedHours);
            const minutes = Math.round((normalizedHours - hours) * 60);
            const totalMinutes = hours * 60 + minutes;
            
            setTime(totalMinutes);
        }
    } catch (error) {
        setToSunriseFallback(targetLat, targetLon, dateStr);
    }
}

// Fallback sunrise using SunCalc
function setToSunriseFallback(targetLat, targetLon, dateStr) {
    const [year, month, day] = dateStr.split('-').map(Number);
    const date = new Date(Date.UTC(year, month - 1, day, 12, 0, 0));
    const times = SunCalc.getTimes(date, targetLat, targetLon);
    const sunrise = times.sunrise;
    
    const utcHours = sunrise.getUTCHours() + sunrise.getUTCMinutes() / 60;
    const solarHours = utcHours + targetLon / 15;
    
    let normalizedSolarHours = solarHours;
    while (normalizedSolarHours < 0) normalizedSolarHours += 24;
    while (normalizedSolarHours >= 24) normalizedSolarHours -= 24;
    
    const hours = Math.floor(normalizedSolarHours);
    const minutes = Math.round((normalizedSolarHours - hours) * 60);
    setTime(hours * 60 + minutes);
}

// Set to sunset using sunrise-sunset.org API with timezone
async function setToSunset() {
    let targetLat, targetLon;
    if (selectedLocation) {
        targetLat = selectedLocation.lat;
        targetLon = selectedLocation.lon;
    } else {
        const center = map.getCenter();
        targetLat = center.lat;
        targetLon = center.lng;
    }
    
    const dateStr = document.getElementById('date-picker').value;
    
    try {
        const response = await fetch(`https://api.sunrise-sunset.org/json?lat=${targetLat}&lng=${targetLon}&date=${dateStr}&formatted=0`);
        const data = await response.json();
        
        if (data.status === 'OK') {
            const sunsetUTC = new Date(data.results.sunset);
            
            // Get local time offset
            const localOffsetMinutes = await getLocationTimezoneOffset(targetLat, targetLon, sunsetUTC);
            
            // Convert UTC to local clock time
            const localHours = sunsetUTC.getUTCHours() + sunsetUTC.getUTCMinutes() / 60 + localOffsetMinutes / 60;
            
            let normalizedHours = localHours;
            while (normalizedHours < 0) normalizedHours += 24;
            while (normalizedHours >= 24) normalizedHours -= 24;
            
            const hours = Math.floor(normalizedHours);
            const minutes = Math.round((normalizedHours - hours) * 60);
            const totalMinutes = hours * 60 + minutes;
            
            setTime(totalMinutes);
        }
    } catch (error) {
        setToSunsetFallback(targetLat, targetLon, dateStr);
    }
}

// Fallback sunset using SunCalc
function setToSunsetFallback(targetLat, targetLon, dateStr) {
    const [year, month, day] = dateStr.split('-').map(Number);
    const date = new Date(Date.UTC(year, month - 1, day, 12, 0, 0));
    const times = SunCalc.getTimes(date, targetLat, targetLon);
    const sunset = times.sunset;
    
    const utcHours = sunset.getUTCHours() + sunset.getUTCMinutes() / 60;
    const solarHours = utcHours + targetLon / 15;
    
    let normalizedSolarHours = solarHours;
    while (normalizedSolarHours < 0) normalizedSolarHours += 24;
    while (normalizedSolarHours >= 24) normalizedSolarHours -= 24;
    
    const hours = Math.floor(normalizedSolarHours);
    const minutes = Math.round((normalizedSolarHours - hours) * 60);
    setTime(hours * 60 + minutes);
}

// Set to current time
function setToCurrentTime() {
    const now = new Date();
    currentDate = now;
    document.getElementById('date-picker').value = now.toISOString().split('T')[0];
    setTime(now.getHours() * 60 + now.getMinutes());
}

// Update solar data display
function updateSolarData() {
    // Use selected location if available, otherwise use map center
    let targetLat, targetLon;
    if (selectedLocation) {
        targetLat = selectedLocation.lat;
        targetLon = selectedLocation.lon;
    } else {
        const center = map.getCenter();
        targetLat = center.lat;
        targetLon = center.lng;
    }
    
    // Get date from picker and create proper UTC date
    const dateStr = document.getElementById('date-picker').value;
    const [year, month, day] = dateStr.split('-').map(Number);
    const hours = Math.floor(currentMinutes / 60);
    const mins = currentMinutes % 60;
    
    // Create date in UTC for SunCalc
    const date = new Date(Date.UTC(year, month - 1, day, hours, mins, 0));
    
    const sunPos = calculateSunPosition(targetLat, targetLon, date);
    const irradiance = calculateSolarIrradiance(sunPos.altitude, date);
    
    // Validate sunPos
    if (!sunPos || isNaN(sunPos.altitude) || isNaN(sunPos.azimuth)) {
        console.error('Invalid sunPos:', sunPos);
        return;
    }
    
    // Validate irradiance
    if (!irradiance || isNaN(irradiance.total)) {
        console.error('Invalid irradiance:', irradiance);
        return;
    }
    
    // Update sidebar display
    document.getElementById('solar-altitude').textContent = `${sunPos.altitude.toFixed(1)}°`;
    document.getElementById('solar-azimuth').textContent = `${sunPos.azimuth.toFixed(1)}°`;
    document.getElementById('solar-irradiance').textContent = `${irradiance.total.toFixed(0)} W/m²`;
    
    // Calculate daylight hours
    try {
        const dayDate = new Date(Date.UTC(year, month - 1, day, 12, 0, 0));
        const times = SunCalc.getTimes(dayDate, targetLat, targetLon);
        const daylightHours = (times.sunset - times.sunrise) / (1000 * 60 * 60);
        document.getElementById('daylight-hours').textContent = `${daylightHours.toFixed(1)} hrs`;
    } catch (e) {
        document.getElementById('daylight-hours').textContent = '-- hrs';
    }
    
    // Update sun status
    const sunStatus = document.getElementById('sun-status');
    if (sunPos.altitude > 0) {
        sunStatus.innerHTML = '<i class="fas fa-sun me-1"></i> Daytime';
        sunStatus.className = 'badge bg-warning text-dark';
    } else {
        sunStatus.innerHTML = '<i class="fas fa-moon me-1"></i> Nighttime';
        sunStatus.className = 'badge bg-dark';
    }
    
    // Update sun intensity overlay
    updateSunIntensityOverlay(sunPos, irradiance);
    
    // Update sun path if visible
    if (document.getElementById('sun-path-overlay').checked) {
        drawSunPath();
    }
    
    // Update panel grid if visible
    drawPanelGrid();
    
    // Update info box if location is selected
    if (selectedLocation) {
        updateInfoBox(selectedLocation.lat, selectedLocation.lon, sunPos, irradiance);
    }
    
    // Update energy displays
    updateEnergyDisplays();
}

// Update info box for selected location
function updateInfoBox(lat, lon, sunPos, irradiance) {
    // Validate inputs - check all properties we need
    if (!sunPos || isNaN(sunPos.altitude) || isNaN(sunPos.azimuth)) {
        console.error('Invalid sunPos in updateInfoBox:', sunPos);
        return;
    }
    if (!irradiance || irradiance.total === undefined || isNaN(irradiance.total)) {
        console.error('Invalid irradiance in updateInfoBox:', irradiance);
        return;
    }
    if (isNaN(lat) || isNaN(lon)) {
        console.error('Invalid coordinates in updateInfoBox:', lat, lon);
        return;
    }
    
    document.getElementById('info-coords').textContent = `${lat.toFixed(4)}°, ${lon.toFixed(4)}°`;
    document.getElementById('info-altitude').textContent = `${sunPos.altitude.toFixed(1)}°`;
    document.getElementById('info-azimuth').textContent = `${sunPos.azimuth.toFixed(1)}°`;
    
    // Calculate optimal solar panel settings
    const panelInfo = calculateOptimalPanelSettings(lat, lon);
    document.getElementById('info-tilt').textContent = `${panelInfo.optimalTilt.toFixed(1)}°`;
    document.getElementById('info-panel-azimuth').textContent = `${panelInfo.optimalAzimuth.toFixed(0)}°`;
    
    // Show current irradiance (updates with time slider)
    document.getElementById('info-irradiance').textContent = `${irradiance.total.toFixed(0)} W/m²`;
    
    // Tilt angle verification
    const tiltVerification = verifyTiltAngle(panelInfo.optimalTilt, lat);
    document.getElementById('info-tilt-range').textContent = `${tiltVerification.minTilt.toFixed(0)}° - ${tiltVerification.maxTilt.toFixed(0)}°`;
    const tiltStatusEl = document.getElementById('info-tilt-status');
    tiltStatusEl.textContent = tiltVerification.status;
    tiltStatusEl.className = `stat-value ${tiltVerification.statusClass}`;
    
    // Update marker popup if exists
    if (marker) {
        marker.setPopupContent(`
            <b>Location:</b> ${lat.toFixed(4)}°, ${lon.toFixed(4)}°<br>
            <b>Sun Altitude:</b> ${sunPos.altitude.toFixed(1)}°<br>
            <b>Sun Azimuth:</b> ${sunPos.azimuth.toFixed(1)}°<br>
            <hr>
            <b>Optimal Tilt:</b> ${panelInfo.optimalTilt.toFixed(1)}°<br>
            <b>Panel Azimuth:</b> ${panelInfo.optimalAzimuth.toFixed(0)}°<br>
            <b>Irradiance:</b> ${irradiance.total.toFixed(0)} W/m²<br>
            <hr>
            <b>Tilt Range:</b> ${tiltVerification.minTilt.toFixed(0)}° - ${tiltVerification.maxTilt.toFixed(0)}°<br>
            <b>Status:</b> ${tiltVerification.status}
        `);
    }
}

// Verify tilt angle and provide acceptable range
function verifyTiltAngle(optimalTilt, lat) {
    // Acceptable tilt range is typically ±15° from optimal
    // This accounts for seasonal variations and practical installation constraints
    const tolerance = 15;
    const minTilt = Math.max(0, optimalTilt - tolerance);
    const maxTilt = Math.min(90, optimalTilt + tolerance);
    
    // Determine status based on latitude
    let status, statusClass;
    const absLat = Math.abs(lat);
    
    if (absLat < 10) {
        // Near equator - nearly horizontal panels
        status = 'Excellent (Equatorial)';
        statusClass = 'text-success';
    } else if (absLat < 25) {
        // Tropical region
        status = 'Excellent (Tropical)';
        statusClass = 'text-success';
    } else if (absLat < 45) {
        // Temperate region
        status = 'Good (Temperate)';
        statusClass = 'text-primary';
    } else if (absLat < 60) {
        // High latitude
        status = 'Moderate (High Lat)';
        statusClass = 'text-warning';
    } else {
        // Polar region - limited solar potential
        status = 'Limited (Polar)';
        statusClass = 'text-danger';
    }
    
    // Check if optimal tilt is within practical limits
    if (optimalTilt > 60) {
        status += ' - Steep Angle';
        statusClass = 'text-warning';
    }
    
    return {
        minTilt: minTilt,
        maxTilt: maxTilt,
        status: status,
        statusClass: statusClass
    };
}

// Calculate energy for a specific month
function calculateMonthlyEnergy(lat, lon, tilt, azimuth, month) {
    const year = 2024;
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    let totalEnergy = 0;
    
    for (let day = 1; day <= daysInMonth; day++) {
        const dayEnergy = calculateDailyEnergyForDate(lat, lon, tilt, azimuth, year, month, day);
        totalEnergy += dayEnergy;
    }
    
    return totalEnergy;
}

// Calculate energy for a specific day
function calculateDailyEnergy(lat, lon, tilt, azimuth, date) {
    const year = date.getFullYear();
    const month = date.getMonth();
    const day = date.getDate();
    return calculateDailyEnergyForDate(lat, lon, tilt, azimuth, year, month, day);
}

// Calculate energy for a specific date (internal function)
function calculateDailyEnergyForDate(lat, lon, tilt, azimuth, year, month, day) {
    const solarConstant = 1367;
    const date = new Date(year, month, day);
    const dayOfYear = Math.floor((date - new Date(year, 0, 0)) / (1000 * 60 * 60 * 24));
    
    // Calculate declination angle
    const declination = 23.45 * Math.sin((360/365) * (dayOfYear - 81) * Math.PI / 180);
    
    // Calculate sunrise/sunset times
    const latRad = lat * Math.PI / 180;
    const declRad = declination * Math.PI / 180;
    
    let cosHourAngle = -Math.tan(latRad) * Math.tan(declRad);
    cosHourAngle = Math.max(-1, Math.min(1, cosHourAngle));
    const hourAngle = Math.acos(cosHourAngle);
    
    const dayLength = 2 * hourAngle * 180 / Math.PI / 15;
    
    if (dayLength <= 0 || isNaN(dayLength)) {
        return 0;
    }
    
    const sinElevation = Math.sin(latRad) * Math.sin(declRad) + 
                        Math.cos(latRad) * Math.cos(declRad) * Math.sin(hourAngle) / hourAngle;
    const avgElevation = Math.asin(Math.max(-1, Math.min(1, sinElevation))) * 180 / Math.PI;
    
    const effectiveElevation = Math.max(avgElevation, 5);
    const avgAirMass = 1 / (Math.sin(effectiveElevation * Math.PI / 180) + 0.50572 * Math.pow(96.07995 - effectiveElevation, -1.6364));
    const atmosphericTransmittance = Math.pow(0.7, Math.pow(avgAirMass, 0.678));
    
    const peakIrradiance = solarConstant * atmosphericTransmittance / 1000;
    const dailyInsolation = peakIrradiance * dayLength * 0.5;
    
    const tiltRad = tilt * Math.PI / 180;
    const latRad2 = lat * Math.PI / 180;
    const declRad2 = declination * Math.PI / 180;
    
    const cosIncidence = 
        Math.sin(declRad2) * (Math.sin(latRad2) * Math.cos(tiltRad) - Math.cos(latRad2) * Math.sin(tiltRad)) +
        Math.cos(declRad2) * Math.cos(latRad2 - tiltRad);
    
    const denominator = Math.cos(latRad2 - declRad2 * 0.5);
    const tiltFactor = denominator !== 0 ? Math.max(0, cosIncidence / denominator) : 1;
    
    const dailyEnergy = dailyInsolation * Math.min(tiltFactor, 1.5) * 0.75;
    return Math.max(0, dailyEnergy);
}

// Calculate optimal solar panel settings for yearly energy generation
function calculateOptimalPanelSettings(lat, lon) {
    // Optimal tilt angle is approximately equal to latitude for maximum yearly energy
    // For Northern Hemisphere: tilt = latitude
    // For Southern Hemisphere: tilt = -latitude (or |latitude| facing North)
    const optimalTilt = Math.abs(lat);
    
    // Optimal azimuth: South (180°) for Northern Hemisphere, North (0°) for Southern Hemisphere
    const optimalAzimuth = lat >= 0 ? 180 : 0;
    
    // Calculate yearly energy generation (simplified model)
    // Uses average daily insolation based on latitude
    const yearlyEnergy = calculateYearlyEnergy(lat, lon, optimalTilt, optimalAzimuth);
    
    return {
        optimalTilt: optimalTilt,
        optimalAzimuth: optimalAzimuth,
        yearlyEnergy: yearlyEnergy
    };
}

// Calculate yearly energy generation for a solar panel
function calculateYearlyEnergy(lat, lon, tilt, azimuth) {
    // Simplified model for yearly energy calculation
    // Based on latitude and panel orientation
    
    let totalEnergy = 0;
    const solarConstant = 1367; // W/m²
    
    // Calculate for each day of the year
    for (let day = 0; day < 365; day++) {
        const date = new Date(2024, 0, day + 1);
        const dayOfYear = day;
        
        // Calculate declination angle (varies through the year)
        const declination = 23.45 * Math.sin((360/365) * (dayOfYear - 81) * Math.PI / 180);
        
        // Calculate sunrise/sunset times
        const latRad = lat * Math.PI / 180;
        const declRad = declination * Math.PI / 180;
        
        // Hour angle at sunrise/sunset
        let cosHourAngle = -Math.tan(latRad) * Math.tan(declRad);
        cosHourAngle = Math.max(-1, Math.min(1, cosHourAngle));
        const hourAngle = Math.acos(cosHourAngle);
        
        // Day length in hours
        const dayLength = 2 * hourAngle * 180 / Math.PI / 15;
        
        // Skip days with no daylight (polar regions)
        if (dayLength <= 0 || isNaN(dayLength)) {
            continue;
        }
        
        // Average solar elevation during the day
        const sinElevation = Math.sin(latRad) * Math.sin(declRad) + 
                            Math.cos(latRad) * Math.cos(declRad) * Math.sin(hourAngle) / hourAngle;
        const avgElevation = Math.asin(Math.max(-1, Math.min(1, sinElevation))) * 180 / Math.PI;
        
        // Atmospheric transmittance (simplified)
        const effectiveElevation = Math.max(avgElevation, 5);
        const avgAirMass = 1 / (Math.sin(effectiveElevation * Math.PI / 180) + 0.50572 * Math.pow(96.07995 - effectiveElevation, -1.6364));
        const atmosphericTransmittance = Math.pow(0.7, Math.pow(avgAirMass, 0.678));
        
        // Daily insolation on horizontal surface (convert W to kW: divide by 1000)
        // Energy = Power × Time, so: (W/m²) × hours = Wh/m², then /1000 = kWh/m²
        const peakIrradiance = solarConstant * atmosphericTransmittance / 1000; // kW/m²
        const dailyInsolation = peakIrradiance * dayLength * 0.5; // kWh/m² (0.5 factor for daily average)
        
        // Apply tilt factor (ratio of insolation on tilted vs horizontal surface)
        const tiltRad = tilt * Math.PI / 180;
        const latRad2 = lat * Math.PI / 180;
        const declRad2 = declination * Math.PI / 180;
        
        // Optimal tilt factor calculation
        const cosIncidence = 
            Math.sin(declRad2) * (Math.sin(latRad2) * Math.cos(tiltRad) - Math.cos(latRad2) * Math.sin(tiltRad)) +
            Math.cos(declRad2) * Math.cos(latRad2 - tiltRad);
        
        const denominator = Math.cos(latRad2 - declRad2 * 0.5);
        const tiltFactor = denominator !== 0 ? Math.max(0, cosIncidence / denominator) : 1;
        
        // Daily energy for this day (75% system efficiency)
        const dailyEnergy = dailyInsolation * Math.min(tiltFactor, 1.5) * 0.75;
        totalEnergy += Math.max(0, dailyEnergy);
    }
    
    return totalEnergy;
}

// Get solar panel configuration from UI
function getPanelConfig() {
    const width = parseFloat(document.getElementById('panel-width').value) || 0;
    const height = parseFloat(document.getElementById('panel-height').value) || 0;
    let rows = parseInt(document.getElementById('grid-rows').value) || 0;
    let cols = parseInt(document.getElementById('grid-cols').value) || 0;
    
    // Limit grid size for performance (max 80x80 = 6400 panels)
    rows = Math.min(80, Math.max(1, rows));
    cols = Math.min(80, Math.max(1, cols));
    
    // Update input fields if values were clamped
    document.getElementById('grid-rows').value = rows;
    document.getElementById('grid-cols').value = cols;
    
    return {
        width: Math.max(0, width),
        height: Math.max(0, height),
        rows: rows,
        cols: cols,
        wattage: parseFloat(document.getElementById('panel-wattage').value) || 400,
        efficiency: parseFloat(document.getElementById('panel-efficiency').value) || 20,
        tilt: parseFloat(document.getElementById('tilt-slider').value) || 34,
        azimuth: parseFloat(document.getElementById('azimuth-slider').value) || 180
    };
}

// Get direction name from azimuth angle
function getAzimuthDirectionName(azimuth) {
    if (azimuth >= 337.5 || azimuth < 22.5) return 'North';
    if (azimuth >= 22.5 && azimuth < 67.5) return 'NE';
    if (azimuth >= 67.5 && azimuth < 112.5) return 'East';
    if (azimuth >= 112.5 && azimuth < 157.5) return 'SE';
    if (azimuth >= 157.5 && azimuth < 202.5) return 'South';
    if (azimuth >= 202.5 && azimuth < 247.5) return 'SW';
    if (azimuth >= 247.5 && azimuth < 292.5) return 'West';
    if (azimuth >= 292.5 && azimuth < 337.5) return 'NW';
    return 'North';
}

// Calculate system summary
function updateSystemSummary() {
    const config = getPanelConfig();
    const totalPanels = config.rows * config.cols;
    const totalArea = (config.width * config.height * totalPanels).toFixed(1);
    const systemCapacity = (config.wattage * totalPanels / 1000).toFixed(1);
    
    document.getElementById('total-panels').textContent = totalPanels;
    document.getElementById('total-area').textContent = `${totalArea} m²`;
    document.getElementById('system-capacity').textContent = `${systemCapacity} kWp`;
    
    return { totalPanels, totalArea, systemCapacity };
}

// Calculate current power output based on sun position
function calculateCurrentPower(lat, tilt) {
    if (!selectedLocation) return 0;
    
    const config = getPanelConfig();
    
    // Return 0 if any dimension is 0
    if (config.width <= 0 || config.height <= 0 || config.rows <= 0 || config.cols <= 0) {
        return 0;
    }
    
    const panelAzimuth = config.azimuth;
    
    const dateStr = document.getElementById('date-picker').value;
    const [year, month, day] = dateStr.split('-').map(Number);
    const date = new Date(Date.UTC(year, month - 1, day, currentMinutes / 60, currentMinutes % 60, 0));
    
    const sunPos = calculateSunPosition(lat, selectedLocation.lon, date);
    
    if (sunPos.altitude <= 0) return 0;
    
    const solarConstant = 1367;
    const altitudeRad = sunPos.altitude * Math.PI / 180;
    const airMass = 1 / (Math.sin(altitudeRad) + 0.50572 * Math.pow(96.07995 - sunPos.altitude, -1.6364));
    const atmosphericTransmittance = Math.pow(0.7, Math.pow(airMass, 0.678));
    
    const irradiance = solarConstant * atmosphericTransmittance * Math.sin(altitudeRad);
    
    const tiltRad = tilt * Math.PI / 180;
    const panelAzimuthRad = panelAzimuth * Math.PI / 180;
    const sunAzimuthRad = sunPos.azimuth * Math.PI / 180;
    
    const cosIncidence = Math.sin(altitudeRad) * Math.cos(tiltRad) + 
                         Math.cos(altitudeRad) * Math.sin(tiltRad) * Math.cos(sunAzimuthRad - panelAzimuthRad);
    
    const incidenceAngle = Math.acos(Math.max(-1, Math.min(1, cosIncidence)));
    
    const effectiveIrradiance = irradiance * Math.cos(incidenceAngle);
    const panelEfficiency = config.efficiency / 100;
    const systemEfficiency = 0.85;
    
    const totalPanels = config.rows * config.cols;
    const panelArea = config.width * config.height;
    const totalArea = totalPanels * panelArea;
    
    const power = (effectiveIrradiance * totalArea * panelEfficiency * systemEfficiency) / 1000;
    
    return Math.max(0, power);
}

// Calculate daily energy for selected date with panel config
function calculateDailyEnergyForPanel(lat, tilt) {
    if (!selectedLocation) return 0;
    
    const config = getPanelConfig();
    
    // Return 0 if any dimension is 0
    if (config.width <= 0 || config.height <= 0 || config.rows <= 0 || config.cols <= 0) {
        return 0;
    }
    
    const panelAzimuth = config.azimuth;
    
    const dateStr = document.getElementById('date-picker').value;
    const [year, month, day] = dateStr.split('-').map(Number);
    
    let totalEnergy = 0;
    
    for (let hour = 0; hour < 24; hour++) {
        for (let min = 0; min < 60; min += 15) {
            const date = new Date(Date.UTC(year, month - 1, day, hour, min, 0));
            const sunPos = calculateSunPosition(lat, selectedLocation.lon, date);
            
            if (sunPos.altitude > 0) {
                const solarConstant = 1367;
                const altitudeRad = sunPos.altitude * Math.PI / 180;
                const airMass = 1 / (Math.sin(altitudeRad) + 0.50572 * Math.pow(96.07995 - sunPos.altitude, -1.6364));
                const atmosphericTransmittance = Math.pow(0.7, Math.pow(airMass, 0.678));
                
                const irradiance = solarConstant * atmosphericTransmittance * Math.sin(altitudeRad);
                
                const tiltRad = tilt * Math.PI / 180;
                const panelAzimuthRad = panelAzimuth * Math.PI / 180;
                const sunAzimuthRad = sunPos.azimuth * Math.PI / 180;
                
                const cosIncidence = Math.sin(altitudeRad) * Math.cos(tiltRad) + 
                                     Math.cos(altitudeRad) * Math.sin(tiltRad) * Math.cos(sunAzimuthRad - panelAzimuthRad);
                
                const incidenceAngle = Math.acos(Math.max(-1, Math.min(1, cosIncidence)));
                
                const effectiveIrradiance = irradiance * Math.cos(incidenceAngle);
                const panelEfficiency = config.efficiency / 100;
                const systemEfficiency = 0.85;
                
                const totalPanels = config.rows * config.cols;
                const panelArea = config.width * config.height;
                const totalArea = totalPanels * panelArea;
                
                const power = (effectiveIrradiance * totalArea * panelEfficiency * systemEfficiency) / 1000;
                totalEnergy += power * (15 / 60);
            }
        }
    }
    
    return Math.max(0, totalEnergy);
}

// Calculate annual energy with custom tilt
function calculateAnnualEnergyWithTilt(lat, tilt) {
    const config = getPanelConfig();
    
    // Return 0 if any dimension is 0
    if (config.width <= 0 || config.height <= 0 || config.rows <= 0 || config.cols <= 0) {
        return 0;
    }
    
    const azimuth = config.azimuth;
    const baseEnergy = calculateYearlyEnergy(lat, selectedLocation ? selectedLocation.lon : 0, tilt, azimuth);
    
    const totalPanels = config.rows * config.cols;
    const panelArea = config.width * config.height;
    const totalArea = totalPanels * panelArea;
    const panelEfficiency = config.efficiency / 100;
    
    const annualEnergy = baseEnergy * totalArea * panelEfficiency * 0.85;
    
    return annualEnergy;
}

// Update all energy displays
function updateEnergyDisplays() {
    if (!selectedLocation) {
        document.getElementById('current-power').textContent = '-- kW';
        document.getElementById('today-energy').textContent = '-- kWh';
        document.getElementById('monthly-energy').textContent = '-- kWh';
        document.getElementById('annual-energy').textContent = '-- kWh';
        return;
    }
    
    const config = getPanelConfig();
    const lat = selectedLocation.lat;
    
    const currentPower = calculateCurrentPower(lat, config.tilt);
    const dailyEnergy = calculateDailyEnergyForPanel(lat, config.tilt);
    const annualEnergy = calculateAnnualEnergyWithTilt(lat, config.tilt);
    const monthlyEnergy = annualEnergy / 12;
    
    document.getElementById('current-power').textContent = `${currentPower.toFixed(2)} kW`;
    document.getElementById('today-energy').textContent = `${dailyEnergy.toFixed(1)} kWh`;
    document.getElementById('monthly-energy').textContent = `${monthlyEnergy.toFixed(0)} kWh`;
    document.getElementById('annual-energy').textContent = `${annualEnergy.toFixed(0)} kWh`;
}

// Initialize solar panel controls
function initPanelControls() {
    const tiltSlider = document.getElementById('tilt-slider');
    const tiltValue = document.getElementById('tilt-value');
    const resetTiltBtn = document.getElementById('reset-tilt');
    
    tiltSlider.addEventListener('input', function() {
        tiltValue.textContent = this.value;
        updateEnergyDisplays();
        drawPanelGrid();
    });
    
    resetTiltBtn.addEventListener('click', function() {
        if (selectedLocation) {
            const optimalTilt = Math.max(1, Math.abs(selectedLocation.lat).toFixed(0));
            tiltSlider.value = optimalTilt;
            tiltValue.textContent = optimalTilt;
            updateEnergyDisplays();
            drawPanelGrid();
        }
    });
    
    const azimuthSlider = document.getElementById('azimuth-slider');
    const azimuthValue = document.getElementById('azimuth-value');
    const azimuthDirection = document.getElementById('azimuth-direction');
    const resetAzimuthBtn = document.getElementById('reset-azimuth');
    
    azimuthSlider.addEventListener('input', function() {
        azimuthValue.textContent = this.value;
        azimuthDirection.textContent = getAzimuthDirectionName(parseFloat(this.value));
        updateEnergyDisplays();
        drawPanelGrid();
    });
    
    resetAzimuthBtn.addEventListener('click', function() {
        if (selectedLocation) {
            const optimalAzimuth = selectedLocation.lat >= 0 ? 180 : 0;
            azimuthSlider.value = optimalAzimuth;
            azimuthValue.textContent = optimalAzimuth;
            azimuthDirection.textContent = getAzimuthDirectionName(optimalAzimuth);
            updateEnergyDisplays();
            drawPanelGrid();
        }
    });
    
    const inputs = ['panel-width', 'panel-height', 'grid-rows', 'grid-cols', 'panel-wattage', 'panel-efficiency'];
    inputs.forEach(id => {
        document.getElementById(id).addEventListener('input', function() {
            updateSystemSummary();
            updateEnergyDisplays();
            drawPanelGrid();
        });
    });
    
    document.getElementById('show-panel-grid').addEventListener('change', function() {
        drawPanelGrid();
    });
    
    document.getElementById('show-tooltips').addEventListener('change', function() {
        drawPanelGrid();
    });
}

// Draw solar panel grid on map
function drawPanelGrid() {
    if (panelGridLayer) {
        map.removeLayer(panelGridLayer);
    }
    
    if (!selectedLocation || !document.getElementById('show-panel-grid').checked) {
        return;
    }
    
    const config = getPanelConfig();
    
    // Don't draw if any dimension is 0
    if (config.width <= 0 || config.height <= 0 || config.rows <= 0 || config.cols <= 0) {
        return;
    }
    
    panelGridLayer = L.layerGroup().addTo(map);
    
    const lat = selectedLocation.lat;
    const lon = selectedLocation.lon;
    const tilt = config.tilt;
    const azimuth = config.azimuth;
    
    const panelWidthM = config.width;
    const panelHeightM = config.height;
    const rows = config.rows;
    const cols = config.cols;
    
    const metersPerDegreeLat = 111320;
    const metersPerDegreeLon = 111320 * Math.cos(lat * Math.PI / 180);
    
    const panelWidthDegLon = panelWidthM / metersPerDegreeLon;
    const panelHeightDegLat = panelHeightM / metersPerDegreeLat;
    
    const gapM = 0.1;
    const gapDegLat = gapM / metersPerDegreeLat;
    const gapDegLon = gapM / metersPerDegreeLon;
    
    const tiltRad = tilt * Math.PI / 180;
    const cosTilt = Math.cos(tiltRad);
    
    const projectedWidthDeg = panelWidthDegLon;
    const projectedHeightDeg = panelHeightDegLat * cosTilt;
    
    const cellSpacingEW = panelWidthDegLon + gapDegLon;
    const cellSpacingNS = panelHeightDegLat + gapDegLat;
    
    const azimuthRad = azimuth * Math.PI / 180;
    const cosAz = Math.cos(azimuthRad);
    const sinAz = Math.sin(azimuthRad);
    
    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            const offsetEW = (col - (cols - 1) / 2) * cellSpacingEW;
            const offsetNS = (row - (rows - 1) / 2) * cellSpacingNS;
            
            const rotatedOffsetEW = offsetEW * cosAz - offsetNS * sinAz;
            const rotatedOffsetNS = offsetEW * sinAz + offsetNS * cosAz;
            
            const panelCenterLat = lat - rotatedOffsetNS;
            const panelCenterLon = lon + rotatedOffsetEW;
            
            const halfWidthEW = projectedWidthDeg / 2;
            const halfHeightNS = projectedHeightDeg / 2;
            
            const cornerOffsets = [
                [-halfHeightNS, -halfWidthEW],
                [-halfHeightNS, halfWidthEW],
                [halfHeightNS, halfWidthEW],
                [halfHeightNS, -halfWidthEW]
            ];
            
            const corners = cornerOffsets.map(([offsetNS, offsetEW]) => {
                const rotEW = offsetEW * cosAz - offsetNS * sinAz;
                const rotNS = offsetEW * sinAz + offsetNS * cosAz;
                return [panelCenterLat - rotNS, panelCenterLon + rotEW];
            });
            
            const panelColor = `hsl(220, 70%, ${25 + tilt * 0.2}%)`;
            
            const panelRect = L.polygon(corners, {
                color: '#1a1a2e',
                weight: 1,
                fillColor: panelColor,
                fillOpacity: 0.85
            });
            
            const showTooltipsPanel = document.getElementById('show-tooltips');
            if (!showTooltipsPanel || showTooltipsPanel.checked) {
                const directionName = getAzimuthDirectionName(azimuth);
                panelRect.bindPopup(`
                    <b>Panel ${row + 1}-${col + 1}</b><br>
                    Size: ${panelWidthM}m × ${panelHeightM}m<br>
                    Tilt: ${tilt}°<br>
                    Azimuth: ${azimuth}° (${directionName})
                `);
            }
            
            panelGridLayer.addLayer(panelRect);
        }
    }
}

// Update current time display
function updateCurrentTimeDisplay() {
    const now = new Date();
    document.getElementById('current-time-display').innerHTML = 
        `<i class="fas fa-clock me-1"></i> ${now.toLocaleTimeString()}`;
}

// Synchronous timezone offset calculation (same logic as async version)
function getTimezoneOffsetSync(lat, lon, date) {
    // Iran (UTC+3:30) - specific to Iran only
    if (lon >= 44 && lon <= 64 && lat >= 25 && lat <= 40) {
        // Exclude UAE and other Gulf states (they use UTC+4)
        if (lon >= 51 && lon <= 57 && lat >= 22 && lat <= 26) {
            return 4;
        }
        // Exclude Turkmenistan (uses UTC+5)
        if (lat >= 35 && lon >= 52 && lon <= 67) {
            return 5;
        }
        return 3.5;
    }
    
    // Pakistan (UTC+5)
    if (lon >= 60 && lon <= 80 && lat >= 23 && lat <= 37) {
        return 5;
    }
    // India (UTC+5:30)
    if (lon >= 68 && lon <= 97 && lat >= 6 && lat <= 35) {
        return 5.5;
    }
    // China (UTC+8) - single timezone for entire country
    if (lon >= 73 && lon <= 135 && lat >= 18 && lat <= 53) {
        // Kyrgyzstan exception (UTC+6)
        if (lon >= 73 && lon <= 80 && lat >= 39 && lat <= 44) {
            return 6;
        }
        return 8;
    }
    // Japan (UTC+9)
    if (lon >= 122 && lon <= 154 && lat >= 24 && lat <= 46) {
        return 9;
    }
    // Australia - multiple timezones
    if (lat < -10 && lon > 110 && lon < 155) {
        if (lon > 140) {
            const month = date.getUTCMonth();
            const isDST = month < 4 || month > 9;
            return isDST ? 11 : 10;
        }
        if (lon > 129 && lon <= 140) {
            const month = date.getUTCMonth();
            const isDST = month < 4 || month > 9;
            return isDST ? 10.5 : 9.5;
        }
        return 8;
    }
    // Canada timezones (more accurate than longitude-based)
    if (lat >= 41 && lat <= 84 && lon >= -141 && lon <= -52) {
        const month = date.getUTCMonth();
        const day = date.getUTCDate();
        const isDST = (month > 2 && month < 10) || (month === 2 && day >= 8) || (month === 10 && day < 7);
        // Pacific: British Columbia (west)
        if (lon >= -141 && lon < -125) return isDST ? -7 : -8;
        // Mountain: Alberta, BC interior, Yukon, Northwest Territories
        if (lon >= -125 && lon < -102) return isDST ? -6 : -7;
        // Central: Saskatchewan, Manitoba, Nunavut (west)
        if (lon >= -102 && lon < -85) return isDST ? -5 : -6;
        // Eastern: Ontario, Quebec, Nunavut (east)
        if (lon >= -85 && lon < -66) return isDST ? -4 : -5;
        // Atlantic: Maritime provinces
        return isDST ? -3 : -4;
    }
    // USA timezones
    if (lat >= 24 && lat <= 50 && lon >= -125 && lon <= -66) {
        const month = date.getUTCMonth();
        const day = date.getUTCDate();
        const isDST = (month > 2 && month < 10) || (month === 2 && day >= 8) || (month === 10 && day < 7);
        if (lon >= -125 && lon < -110) return isDST ? -7 : -8;
        if (lon >= -110 && lon < -100) return isDST ? -6 : -7;
        if (lon >= -100 && lon < -85) return isDST ? -5 : -6;
        return isDST ? -4 : -5;
    }
    // Europe
    if (lat >= 35 && lat <= 70 && lon >= -10 && lon <= 40) {
        const month = date.getUTCMonth();
        const isDST = month > 2 && month < 10;
        const baseOffset = Math.round(lon / 15);
        return baseOffset + (isDST ? 1 : 0);
    }
    // Default: longitude-based with DST
    const lonOffset = Math.round(lon / 15);
    const month = date.getUTCMonth();
    const day = date.getUTCDate();
    let isDST = false;
    if (lat > 23.5) {
        isDST = (month > 2 && month < 10) || (month === 2 && day >= 8) || (month === 10 && day < 7);
    } else if (lat < -23.5) {
        isDST = month < 4 || month > 9;
    }
    return lonOffset + (isDST ? 1 : 0);
}

// Calculate sun position
function calculateSunPosition(lat, lon, date) {
    // Validate inputs
    if (isNaN(lat) || isNaN(lon)) {
        console.error('Invalid lat/lon:', lat, lon);
        return { altitude: 0, azimuth: 0 };
    }
    
    if (typeof SunCalc === 'undefined') {
        console.error('SunCalc not loaded');
        return { altitude: 45, azimuth: 180 };
    }
    
    // The slider shows local clock time (timezone-based)
    // We need to convert to UTC for SunCalc
    
    // Get timezone offset in hours
    const tzOffset = getTimezoneOffsetSync(lat, lon, date);
    
    // The slider shows local time, stored as UTC in the date object
    // To get actual UTC for SunCalc, we subtract the timezone offset
    const localHours = date.getUTCHours() + date.getUTCMinutes() / 60;
    const actualUtcHours = localHours - tzOffset;
    
    // Create the actual UTC date for SunCalc
    const utcDate = new Date(date);
    const totalMinutes = Math.round(actualUtcHours * 60);
    const finalHours = Math.floor(totalMinutes / 60);
    const finalMinutes = totalMinutes % 60;
    
    utcDate.setUTCHours(finalHours, finalMinutes, 0, 0);
    
    // Handle day overflow
    if (totalMinutes < 0) {
        utcDate.setUTCDate(utcDate.getUTCDate() - 1);
    } else if (totalMinutes >= 1440) {
        utcDate.setUTCDate(utcDate.getUTCDate() + 1);
    }
    
    const position = SunCalc.getPosition(utcDate, lat, lon);
    const altitudeDeg = position.altitude * (180 / Math.PI);
    const azimuthDeg = position.azimuth * (180 / Math.PI);
    const normalizedAzimuth = (azimuthDeg + 180) % 360;
    return {
        altitude: altitudeDeg,
        azimuth: normalizedAzimuth
    };
}

// Calculate solar irradiance
function calculateSolarIrradiance(altitudeDeg, date = new Date()) {
    // Handle invalid altitude
    if (isNaN(altitudeDeg) || altitudeDeg === null || altitudeDeg === undefined) {
        return { total: 0, direct: 0, diffuse: 0 };
    }
    
    if (altitudeDeg <= 0) {
        return { total: 0, direct: 0, diffuse: 0 };
    }
    
    const altitudeRad = altitudeDeg * Math.PI / 180;
    const airMass = 1 / (Math.sin(altitudeRad) + 0.50572 * Math.pow(96.07995 - altitudeDeg, -1.6364));
    const atmosphericTransmittance = Math.pow(0.7, Math.pow(airMass, 0.678));
    const solarConstant = 1367;
    const directNormal = solarConstant * atmosphericTransmittance * Math.sin(altitudeRad);
    const diffuse = directNormal * 0.1;
    const total = directNormal + diffuse;
    return { total, direct: directNormal, diffuse };
}

// Update sun intensity overlay - only draws when location is selected
function updateSunIntensityOverlay(sunPos, irradiance) {
    if (!sunIntensityLayer) return;
    
    // Only draw sun line if we have a selected location and the checkbox is checked
    const preciseMode = document.getElementById('precise-location-mode');
    if (!selectedLocation || !preciseMode || !preciseMode.checked) {
        sunIntensityLayer.clearLayers();
        return;
    }
    
    // Validate sun position
    if (!sunPos || isNaN(sunPos.altitude) || isNaN(sunPos.azimuth)) {
        console.error('Invalid sun position:', sunPos);
        return;
    }
    
    // Clear and redraw sun direction indicator from selected location
    sunIntensityLayer.clearLayers();
    
    const lat = selectedLocation.lat;
    const lon = selectedLocation.lon;
    
    // Validate coordinates
    if (isNaN(lat) || isNaN(lon)) {
        console.error('Invalid selected location:', lat, lon);
        return;
    }
    
    const sunDistance = 0.05; // degrees
    
    // Add a circle around the selected location
    const locationCircle = L.circle([lat, lon], {
        radius: 100, // 100 meters
        color: '#0d6efd',
        fillColor: '#0d6efd',
        fillOpacity: 0.2,
        weight: 4
    });
    sunIntensityLayer.addLayer(locationCircle);
    
    // Calculate sun position on the horizon (even if below)
    // Azimuth: 0°=North, 90°=East, 180°=South, 270°=West
    // Line should point FROM observer TO the sun
    const displayAltitude = Math.max(sunPos.altitude, 0);
    const azimuthRad = sunPos.azimuth * Math.PI / 180;
    const sunLat = lat + sunDistance * Math.cos(displayAltitude * Math.PI / 180) * Math.cos(azimuthRad);
    const sunLon = lon + sunDistance * Math.cos(displayAltitude * Math.PI / 180) * Math.sin(azimuthRad);
    
    // Validate calculated sun position
    if (isNaN(sunLat) || isNaN(sunLon)) {
        console.error('Invalid calculated sun position:', sunLat, sunLon);
        return;
    }
    
    // Sun direction line - different color for day/night
    const isNight = sunPos.altitude <= 0;
    const sunLine = L.polyline([[lat, lon], [sunLat, sunLon]], {
        color: isNight ? '#6c757d' : '#f39c12',
        weight: 5,
        opacity: isNight ? 0.5 : 0.8,
        dashArray: '10, 5'
    });
    
    sunIntensityLayer.addLayer(sunLine);
    
    // Sun position marker - different style for day/night
    const sunMarker = L.circleMarker([sunLat, sunLon], {
        radius: isNight ? 12 : 15,
        fillColor: isNight ? '#6c757d' : '#f1c40f',
        color: isNight ? '#495057' : '#f39c12',
        weight: 3,
        fillOpacity: isNight ? 0.5 : 0.8
    });
    sunIntensityLayer.addLayer(sunMarker);
}

// Draw sun path for the day
function drawSunPath() {
    if (sunPathLayer) {
        map.removeLayer(sunPathLayer);
    }
    
    sunPathLayer = L.layerGroup().addTo(map);
    
    // Use selected location if available, otherwise use map center
    let targetLat, targetLon;
    if (selectedLocation) {
        targetLat = selectedLocation.lat;
        targetLon = selectedLocation.lon;
    } else {
        const center = map.getCenter();
        targetLat = center.lat;
        targetLon = center.lng;
    }
    
    // Get date from picker
    const dateStr = document.getElementById('date-picker').value;
    const [year, month, day] = dateStr.split('-').map(Number);
    const pathPoints = [];
    
    // Calculate sun position every 30 minutes
    for (let hour = 0; hour < 24; hour++) {
        for (let min = 0; min < 60; min += 30) {
            // Create date in UTC representing local solar time
            const testDate = new Date(Date.UTC(year, month - 1, day, hour, min, 0));
            const sunPos = calculateSunPosition(targetLat, targetLon, testDate);
            
            if (sunPos.altitude > 0) {
                const sunDistance = 0.03;
                const azimuthRad = sunPos.azimuth * Math.PI / 180;
                const sunLat = targetLat + sunDistance * Math.cos(sunPos.altitude * Math.PI / 180) * Math.cos(azimuthRad);
                const sunLon = targetLon + sunDistance * Math.cos(sunPos.altitude * Math.PI / 180) * Math.sin(azimuthRad);
                pathPoints.push([sunLat, sunLon]);
            }
        }
    }
    
    if (pathPoints.length > 1) {
        const pathLine = L.polyline(pathPoints, {
            color: '#f39c12',
            weight: 4,
            opacity: 0.6,
            dashArray: '5, 5'
        });
        sunPathLayer.addLayer(pathLine);
    }
}

// Start the application
document.addEventListener('DOMContentLoaded', initApp);
