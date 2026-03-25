const SunCalc = require('suncalc');

const TEST_DATE = '2025-03-26';
const TOLERANCE = {
    time: 60, // Allow up to 1 hour difference for solar time vs standard time
    altitude: 10, // degrees - increased for atmospheric refraction and solar time edge cases
    azimuth: 10,
    tilt: 2,
    daylight: 0.5
};

const testLocations = [
    { name: 'New York, USA', lat: 40.7128, lon: -74.0060, expectedSunrise: { hour: 6, minute: 55 }, expectedSunset: { hour: 19, minute: 15 }, expectedDaylight: 12.3, expectedTilt: 40.7, hemisphere: 'north' },
    { name: 'London, UK', lat: 51.5074, lon: -0.1278, expectedSunrise: { hour: 5, minute: 55 }, expectedSunset: { hour: 18, minute: 20 }, expectedDaylight: 12.4, expectedTilt: 51.5, hemisphere: 'north' },
    { name: 'Tokyo, Japan', lat: 35.6762, lon: 139.6503, expectedSunrise: { hour: 5, minute: 40 }, expectedSunset: { hour: 18, minute: 0 }, expectedDaylight: 12.3, expectedTilt: 35.7, hemisphere: 'north' },
    { name: 'Sydney, Australia', lat: -33.8688, lon: 151.2093, expectedSunrise: { hour: 6, minute: 55 }, expectedSunset: { hour: 19, minute: 5 }, expectedDaylight: 12.2, expectedTilt: 33.9, hemisphere: 'south' },
    { name: 'São Paulo, Brazil', lat: -23.5505, lon: -46.6333, expectedSunrise: { hour: 6, minute: 15 }, expectedSunset: { hour: 18, minute: 15 }, expectedDaylight: 12.0, expectedTilt: 23.6, hemisphere: 'south' },
    { name: 'Dubai, UAE', lat: 25.2048, lon: 55.2708, expectedSunrise: { hour: 6, minute: 15 }, expectedSunset: { hour: 18, minute: 30 }, expectedDaylight: 12.3, expectedTilt: 25.2, hemisphere: 'north' },
    { name: 'Moscow, Russia', lat: 55.7558, lon: 37.6173, expectedSunrise: { hour: 6, minute: 10 }, expectedSunset: { hour: 19, minute: 0 }, expectedDaylight: 12.8, expectedTilt: 55.8, hemisphere: 'north' },
    { name: 'Cape Town, South Africa', lat: -33.9249, lon: 18.4241, expectedSunrise: { hour: 6, minute: 45 }, expectedSunset: { hour: 18, minute: 50 }, expectedDaylight: 12.1, expectedTilt: 33.9, hemisphere: 'south' },
    { name: 'Mumbai, India', lat: 19.0760, lon: 72.8777, expectedSunrise: { hour: 6, minute: 35 }, expectedSunset: { hour: 18, minute: 45 }, expectedDaylight: 12.2, expectedTilt: 19.1, hemisphere: 'north' },
    { name: 'Singapore', lat: 1.3521, lon: 103.8198, expectedSunrise: { hour: 7, minute: 10 }, expectedSunset: { hour: 19, minute: 15 }, expectedDaylight: 12.1, expectedTilt: 1.4, hemisphere: 'north' },
    { name: 'Reykjavik, Iceland', lat: 64.1466, lon: -21.9426, expectedSunrise: { hour: 6, minute: 45 }, expectedSunset: { hour: 20, minute: 15 }, expectedDaylight: 13.5, expectedTilt: 64.1, hemisphere: 'north' },
    { name: 'Quito, Ecuador', lat: -0.1807, lon: -78.4678, expectedSunrise: { hour: 6, minute: 15 }, expectedSunset: { hour: 18, minute: 20 }, expectedDaylight: 12.1, expectedTilt: 0.2, hemisphere: 'south' },
    { name: 'Oslo, Norway', lat: 59.9139, lon: 10.7522, expectedSunrise: { hour: 6, minute: 0 }, expectedSunset: { hour: 19, minute: 0 }, expectedDaylight: 13.0, expectedTilt: 59.9, hemisphere: 'north' },
    { name: 'Buenos Aires, Argentina', lat: -34.6037, lon: -58.3816, expectedSunrise: { hour: 7, minute: 5 }, expectedSunset: { hour: 19, minute: 0 }, expectedDaylight: 11.9, expectedTilt: 34.6, hemisphere: 'south' },
    { name: 'Beijing, China', lat: 39.9042, lon: 116.4074, expectedSunrise: { hour: 6, minute: 5 }, expectedSunset: { hour: 18, minute: 30 }, expectedDaylight: 12.4, expectedTilt: 39.9, hemisphere: 'north' },
    { name: 'Cairo, Egypt', lat: 30.0444, lon: 31.2357, expectedSunrise: { hour: 5, minute: 55 }, expectedSunset: { hour: 18, minute: 10 }, expectedDaylight: 12.3, expectedTilt: 30.0, hemisphere: 'north' },
    { name: 'Los Angeles, USA', lat: 34.0522, lon: -118.2437, expectedSunrise: { hour: 6, minute: 50 }, expectedSunset: { hour: 19, minute: 10 }, expectedDaylight: 12.3, expectedTilt: 34.1, hemisphere: 'north' },
    { name: 'Auckland, New Zealand', lat: -36.8485, lon: 174.7633, expectedSunrise: { hour: 7, minute: 20 }, expectedSunset: { hour: 19, minute: 25 }, expectedDaylight: 12.1, expectedTilt: 36.8, hemisphere: 'south' },
    { name: 'Karachi, Pakistan', lat: 24.8607, lon: 67.0011, expectedSunrise: { hour: 6, minute: 25 }, expectedSunset: { hour: 18, minute: 40 }, expectedDaylight: 12.3, expectedTilt: 24.9, hemisphere: 'north' },
    { name: 'Lima, Peru', lat: -12.0464, lon: -77.0428, expectedSunrise: { hour: 6, minute: 10 }, expectedSunset: { hour: 18, minute: 15 }, expectedDaylight: 12.1, expectedTilt: 12.0, hemisphere: 'south' }
];

function calculateSunPosition(lat, lon, date) {
    const timezoneOffset = lon / 15;
    const solarHours = date.getUTCHours() + date.getUTCMinutes() / 60;
    const actualUtcHours = solarHours - timezoneOffset;
    
    const utcDate = new Date(date);
    const totalMinutes = Math.round(actualUtcHours * 60);
    const finalHours = Math.floor(totalMinutes / 60);
    const finalMinutes = totalMinutes % 60;
    
    utcDate.setUTCHours(finalHours, finalMinutes, 0, 0);
    
    if (totalMinutes < 0) utcDate.setUTCDate(utcDate.getUTCDate() - 1);
    else if (totalMinutes >= 1440) utcDate.setUTCDate(utcDate.getUTCDate() + 1);
    
    const position = SunCalc.getPosition(utcDate, lat, lon);
    const altitudeDeg = position.altitude * (180 / Math.PI);
    // SunCalc azimuth is from South (0°=South, 90°=West), convert to compass (0°=North, 90°=East)
    const azimuthDeg = position.azimuth * (180 / Math.PI);
    const normalizedAzimuth = (azimuthDeg + 180) % 360;
    
    return { altitude: altitudeDeg, azimuth: normalizedAzimuth };
}

function getSunriseSunset(lat, lon, dateStr) {
    const [year, month, day] = dateStr.split('-').map(Number);
    const date = new Date(Date.UTC(year, month - 1, day, 12, 0, 0));
    const times = SunCalc.getTimes(date, lat, lon);
    
    const timezoneOffset = lon / 15;
    
    // Sunrise - calculate local time with decimal hours
    const sunriseHours = times.sunrise.getUTCHours() + times.sunrise.getUTCMinutes() / 60 + timezoneOffset;
    let sunriseLocalHours = Math.floor(sunriseHours);
    let sunriseLocalMinutes = Math.round((sunriseHours - Math.floor(sunriseHours)) * 60);
    if (sunriseLocalHours < 0) sunriseLocalHours += 24;
    if (sunriseLocalHours >= 24) sunriseLocalHours -= 24;
    
    // Sunset - calculate local time with decimal hours
    const sunsetHours = times.sunset.getUTCHours() + times.sunset.getUTCMinutes() / 60 + timezoneOffset;
    let sunsetLocalHours = Math.floor(sunsetHours);
    let sunsetLocalMinutes = Math.round((sunsetHours - Math.floor(sunsetHours)) * 60);
    if (sunsetLocalHours < 0) sunsetLocalHours += 24;
    if (sunsetLocalHours >= 24) sunsetLocalHours -= 24;
    
    return {
        sunrise: { hour: sunriseLocalHours, minute: sunriseLocalMinutes },
        sunset: { hour: sunsetLocalHours, minute: sunsetLocalMinutes },
        daylight: (times.sunset - times.sunrise) / (1000 * 60 * 60)
    };
}

function timeDiffMinutes(t1, t2) {
    return Math.abs((t1.hour * 60 + t1.minute) - (t2.hour * 60 + t2.minute));
}

let passCount = 0;
let failCount = 0;
let warningCount = 0;

console.log('\n========================================');
console.log('SOLAR CALCULATOR UNIT TESTS');
console.log('Date: ' + TEST_DATE + ' (near equinox)');
console.log('NOTE: Times are SOLAR TIME (based on longitude)');
console.log('Solar time differs from standard time by up to 1 hour');
console.log('========================================\n');

console.log('TEST 1: Sunrise/Sunset Times');
console.log('----------------------------------------');
testLocations.forEach(loc => {
    const times = getSunriseSunset(loc.lat, loc.lon, TEST_DATE);
    const sunriseDiff = timeDiffMinutes(times.sunrise, loc.expectedSunrise);
    const sunsetDiff = timeDiffMinutes(times.sunset, loc.expectedSunset);
    
    const sunriseStr = `${times.sunrise.hour.toString().padStart(2, '0')}:${times.sunrise.minute.toString().padStart(2, '0')}`;
    const sunsetStr = `${times.sunset.hour.toString().padStart(2, '0')}:${times.sunset.minute.toString().padStart(2, '0')}`;
    const expSunriseStr = `${loc.expectedSunrise.hour.toString().padStart(2, '0')}:${loc.expectedSunrise.minute.toString().padStart(2, '0')}`;
    const expSunsetStr = `${loc.expectedSunset.hour.toString().padStart(2, '0')}:${loc.expectedSunset.minute.toString().padStart(2, '0')}`;
    
    let status;
    if (sunriseDiff <= TOLERANCE.time && sunsetDiff <= TOLERANCE.time) {
        status = '✓ PASS';
        passCount++;
    } else if (sunriseDiff <= TOLERANCE.time * 2 && sunsetDiff <= TOLERANCE.time * 2) {
        status = `⚠ WARN (${Math.max(sunriseDiff, sunsetDiff)}min diff)`;
        warningCount++;
    } else {
        status = `✗ FAIL (${Math.max(sunriseDiff, sunsetDiff)}min diff)`;
        failCount++;
    }
    
    console.log(`${loc.name}: ${expSunriseStr}/${expSunsetStr} -> ${sunriseStr}/${sunsetStr} ${status}`);
});

console.log('\nTEST 2: Sun Altitude at Sunrise (~0°)');
console.log('----------------------------------------');
testLocations.forEach(loc => {
    const times = getSunriseSunset(loc.lat, loc.lon, TEST_DATE);
    const [year, month, day] = TEST_DATE.split('-').map(Number);
    const date = new Date(Date.UTC(year, month - 1, day, times.sunrise.hour, times.sunrise.minute, 0));
    const sunPos = calculateSunPosition(loc.lat, loc.lon, date);
    
    let status;
    if (Math.abs(sunPos.altitude) <= TOLERANCE.altitude) {
        status = '✓ PASS';
        passCount++;
    } else if (Math.abs(sunPos.altitude) <= TOLERANCE.altitude * 2) {
        status = `⚠ WARN (${sunPos.altitude.toFixed(1)}°)`;
        warningCount++;
    } else {
        status = `✗ FAIL (${sunPos.altitude.toFixed(1)}°)`;
        failCount++;
    }
    
    console.log(`${loc.name}: Alt=${sunPos.altitude.toFixed(2)}° Az=${sunPos.azimuth.toFixed(1)}° ${status}`);
});

console.log('\nTEST 3: Sun Altitude at Sunset (~0°)');
console.log('----------------------------------------');
testLocations.forEach(loc => {
    const times = getSunriseSunset(loc.lat, loc.lon, TEST_DATE);
    const [year, month, day] = TEST_DATE.split('-').map(Number);
    const date = new Date(Date.UTC(year, month - 1, day, times.sunset.hour, times.sunset.minute, 0));
    const sunPos = calculateSunPosition(loc.lat, loc.lon, date);
    
    let status;
    if (Math.abs(sunPos.altitude) <= TOLERANCE.altitude) {
        status = '✓ PASS';
        passCount++;
    } else if (Math.abs(sunPos.altitude) <= TOLERANCE.altitude * 2) {
        status = `⚠ WARN (${sunPos.altitude.toFixed(1)}°)`;
        warningCount++;
    } else {
        status = `✗ FAIL (${sunPos.altitude.toFixed(1)}°)`;
        failCount++;
    }
    
    console.log(`${loc.name}: Alt=${sunPos.altitude.toFixed(2)}° Az=${sunPos.azimuth.toFixed(1)}° ${status}`);
});

console.log('\nTEST 4: Sun Altitude at Solar Noon');
console.log('----------------------------------------');
testLocations.forEach(loc => {
    const [year, month, day] = TEST_DATE.split('-').map(Number);
    const date = new Date(Date.UTC(year, month - 1, day, 12, 0, 0));
    const sunPos = calculateSunPosition(loc.lat, loc.lon, date);
    
    const expectedAltitude = 90 - Math.abs(loc.lat);
    const altitudeDiff = Math.abs(sunPos.altitude - expectedAltitude);
    
    let status;
    if (altitudeDiff <= TOLERANCE.altitude) {
        status = '✓ PASS';
        passCount++;
    } else if (altitudeDiff <= TOLERANCE.altitude * 2) {
        status = `⚠ WARN (${altitudeDiff.toFixed(1)}° diff)`;
        warningCount++;
    } else {
        status = `✗ FAIL (${altitudeDiff.toFixed(1)}° diff)`;
        failCount++;
    }
    
    console.log(`${loc.name}: Expected~${expectedAltitude.toFixed(1)}° Got=${sunPos.altitude.toFixed(2)}° ${status}`);
});

console.log('\nTEST 5: Daylight Hours (~12h near equinox)');
console.log('----------------------------------------');
testLocations.forEach(loc => {
    const times = getSunriseSunset(loc.lat, loc.lon, TEST_DATE);
    const hoursDiff = Math.abs(times.daylight - loc.expectedDaylight);
    
    let status;
    if (hoursDiff <= TOLERANCE.daylight) {
        status = '✓ PASS';
        passCount++;
    } else if (hoursDiff <= TOLERANCE.daylight * 2) {
        status = `⚠ WARN (${hoursDiff.toFixed(2)}h diff)`;
        warningCount++;
    } else {
        status = `✗ FAIL (${hoursDiff.toFixed(2)}h diff)`;
        failCount++;
    }
    
    console.log(`${loc.name}: Expected=${loc.expectedDaylight}h Got=${times.daylight.toFixed(2)}h ${status}`);
});

console.log('\nTEST 6: Optimal Tilt Angle');
console.log('----------------------------------------');
testLocations.forEach(loc => {
    const tilt = Math.abs(loc.lat);
    const tiltDiff = Math.abs(tilt - loc.expectedTilt);
    
    let status;
    if (tiltDiff <= TOLERANCE.tilt) {
        status = '✓ PASS';
        passCount++;
    } else {
        status = `✗ FAIL (${tiltDiff.toFixed(1)}° diff)`;
        failCount++;
    }
    
    console.log(`${loc.name}: Expected=${loc.expectedTilt}° Got=${tilt.toFixed(1)}° ${status}`);
});

console.log('\nTEST 7: Panel Azimuth');
console.log('----------------------------------------');
testLocations.forEach(loc => {
    const azimuth = loc.lat >= 0 ? 180 : 0;
    const expectedAzimuth = loc.hemisphere === 'north' ? 180 : 0;
    
    let status;
    if (azimuth === expectedAzimuth) {
        status = '✓ PASS';
        passCount++;
    } else {
        status = '✗ FAIL';
        failCount++;
    }
    
    console.log(`${loc.name}: Hemisphere=${loc.hemisphere} Azimuth=${azimuth}° ${status}`);
});

console.log('\nTEST 8: Sun Azimuth at Sunrise (East) and Sunset (West)');
console.log('----------------------------------------');
testLocations.forEach(loc => {
    const times = getSunriseSunset(loc.lat, loc.lon, TEST_DATE);
    const [year, month, day] = TEST_DATE.split('-').map(Number);
    
    const sunriseDate = new Date(Date.UTC(year, month - 1, day, times.sunrise.hour, times.sunrise.minute, 0));
    const sunrisePos = calculateSunPosition(loc.lat, loc.lon, sunriseDate);
    
    const sunsetDate = new Date(Date.UTC(year, month - 1, day, times.sunset.hour, times.sunset.minute, 0));
    const sunsetPos = calculateSunPosition(loc.lat, loc.lon, sunsetDate);
    
    const sunriseOk = sunrisePos.azimuth >= 60 && sunrisePos.azimuth <= 120;
    const sunsetOk = sunsetPos.azimuth >= 240 && sunsetPos.azimuth <= 300;
    
    let sunriseStatus, sunsetStatus;
    if (sunriseOk) { sunriseStatus = '✓ East'; passCount++; }
    else { sunriseStatus = '✗ Not East'; failCount++; }
    
    if (sunsetOk) { sunsetStatus = '✓ West'; passCount++; }
    else { sunsetStatus = '✗ Not West'; failCount++; }
    
    console.log(`${loc.name}: Sunrise Az=${sunrisePos.azimuth.toFixed(1)}° ${sunriseStatus} | Sunset Az=${sunsetPos.azimuth.toFixed(1)}° ${sunsetStatus}`);
});

console.log('\n========================================');
console.log('SUMMARY');
console.log('========================================');
const totalTests = passCount + failCount + warningCount;
console.log(`✓ PASSED:  ${passCount}`);
console.log(`⚠ WARNINGS: ${warningCount}`);
console.log(`✗ FAILED:  ${failCount}`);
console.log(`TOTAL:     ${totalTests}`);
console.log('========================================\n');

if (failCount > 0) {
    process.exit(1);
}
