const SunCalc = require('suncalc');
const https = require('https');

const TEST_DATE = '2025-03-26';
const TOLERANCE_MINUTES = 5;

function getTimezoneOffset(lat, lon, date) {
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
    if (lon >= 60 && lon <= 80 && lat >= 23 && lat <= 37) {
        return 5;
    }
    if (lon >= 68 && lon <= 97 && lat >= 6 && lat <= 35) {
        return 5.5;
    }
    if (lon >= 73 && lon <= 135 && lat >= 18 && lat <= 53) {
        if (lon >= 73 && lon <= 80 && lat >= 39 && lat <= 44) {
            return 6;
        }
        return 8;
    }
    if (lon >= 122 && lon <= 154 && lat >= 24 && lat <= 46) {
        return 9;
    }
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
    if (lat >= 24 && lat <= 50 && lon >= -125 && lon <= -66) {
        const month = date.getUTCMonth();
        const day = date.getUTCDate();
        const isDST = (month > 2 && month < 10) || (month === 2 && day >= 8) || (month === 10 && day < 7);
        if (lon >= -125 && lon < -110) return isDST ? -7 : -8;
        if (lon >= -110 && lon < -100) return isDST ? -6 : -7;
        if (lon >= -100 && lon < -85) return isDST ? -5 : -6;
        return isDST ? -4 : -5;
    }
    // Canada timezones
    if (lat >= 41 && lat <= 84 && lon >= -141 && lon <= -52) {
        const month = date.getUTCMonth();
        const day = date.getUTCDate();
        const isDST = (month > 2 && month < 10) || (month === 2 && day >= 8) || (month === 10 && day < 7);
        if (lon >= -141 && lon < -125) return isDST ? -7 : -8;
        if (lon >= -125 && lon < -102) return isDST ? -6 : -7;
        if (lon >= -102 && lon < -85) return isDST ? -5 : -6;
        if (lon >= -85 && lon < -66) return isDST ? -4 : -5;
        return isDST ? -3 : -4;
    }
    if (lat >= 35 && lat <= 70 && lon >= -10 && lon <= 40) {
        const month = date.getUTCMonth();
        const isDST = month > 2 && month < 10;
        const baseOffset = Math.round(lon / 15);
        return baseOffset + (isDST ? 1 : 0);
    }
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

function fetchAPI(url) {
    return new Promise((resolve, reject) => {
        https.get(url, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    resolve(JSON.parse(data));
                } catch (e) {
                    reject(e);
                }
            });
        }).on('error', reject);
    });
}

async function getGroundTruthSunriseSunset(lat, lon, dateStr) {
    const url = `https://api.sunrise-sunset.org/json?lat=${lat}&lng=${lon}&date=${dateStr}&formatted=0`;
    try {
        const data = await fetchAPI(url);
        if (data.status === 'OK') {
            const sunriseUTC = new Date(data.results.sunrise);
            const sunsetUTC = new Date(data.results.sunset);
            return { sunriseUTC, sunsetUTC, success: true };
        }
    } catch (e) {
        console.error('API error:', e.message);
    }
    return { success: false };
}

function getCalculatedSunriseSunset(lat, lon, dateStr) {
    const [year, month, day] = dateStr.split('-').map(Number);
    const date = new Date(Date.UTC(year, month - 1, day, 12, 0, 0));
    const times = SunCalc.getTimes(date, lat, lon);
    return {
        sunriseUTC: times.sunrise,
        sunsetUTC: times.sunset
    };
}

function formatTime(utcDate, tzOffset) {
    const localMinutes = utcDate.getUTCHours() * 60 + utcDate.getUTCMinutes() + tzOffset * 60;
    let hours = Math.floor(localMinutes / 60);
    let mins = Math.round(localMinutes % 60);
    while (hours < 0) hours += 24;
    while (hours >= 24) hours -= 24;
    while (mins < 0) mins += 60;
    while (mins >= 60) mins -= 60;
    return `${hours.toString().padStart(2, '0')}:${Math.abs(mins).toString().padStart(2, '0')}`;
}

function timeDiffMinutes(date1, date2) {
    return Math.abs((date1 - date2) / (1000 * 60));
}

const testLocations = [
    { name: 'Calgary, Canada', lat: 51.0447, lon: -114.0719 },
    { name: 'New York, USA', lat: 40.7128, lon: -74.0060 },
    { name: 'London, UK', lat: 51.5074, lon: -0.1278 },
    { name: 'Tokyo, Japan', lat: 35.6762, lon: 139.6503 },
    { name: 'Sydney, Australia', lat: -33.8688, lon: 151.2093 },
    { name: 'Islamabad, Pakistan', lat: 33.6844, lon: 73.0479 },
    { name: 'New Delhi, India', lat: 28.6139, lon: 77.2090 },
    { name: 'Beijing, China', lat: 39.9042, lon: 116.4074 },
    { name: 'Tehran, Iran', lat: 35.6892, lon: 51.3890 },
    { name: 'Paris, France', lat: 48.8566, lon: 2.3522 },
    { name: 'Dubai, UAE', lat: 25.2048, lon: 55.2708 },
    { name: 'Singapore', lat: 1.3521, lon: 103.8198 },
    { name: 'São Paulo, Brazil', lat: -23.5505, lon: -46.6333 },
    { name: 'Cape Town, South Africa', lat: -33.9249, lon: 18.4241 },
    { name: 'Moscow, Russia', lat: 55.7558, lon: 37.6173 },
    { name: 'Los Angeles, USA', lat: 34.0522, lon: -118.2437 }
];

async function runTests() {
    console.log('\n========================================');
    console.log('GROUND TRUTH COMPARISON TEST');
    console.log('Comparing with sunrise-sunset.org API');
    console.log(`Date: ${TEST_DATE}`);
    console.log(`Tolerance: ${TOLERANCE_MINUTES} minutes`);
    console.log('========================================\n');
    
    let passCount = 0;
    let failCount = 0;
    let totalTests = 0;
    
    for (const loc of testLocations) {
        console.log(`\n${loc.name} (${loc.lat.toFixed(2)}°, ${loc.lon.toFixed(2)}°)`);
        console.log('-'.repeat(50));
        
        const [year, month, day] = TEST_DATE.split('-').map(Number);
        const date = new Date(Date.UTC(year, month - 1, day, 12, 0, 0));
        const tzOffset = getTimezoneOffset(loc.lat, loc.lon, date);
        
        const groundTruth = await getGroundTruthSunriseSunset(loc.lat, loc.lon, TEST_DATE);
        const calculated = getCalculatedSunriseSunset(loc.lat, loc.lon, TEST_DATE);
        
        if (!groundTruth.success) {
            console.log('  ⚠ Could not fetch ground truth data');
            continue;
        }
        
        const sunriseDiff = timeDiffMinutes(groundTruth.sunriseUTC, calculated.sunriseUTC);
        const sunsetDiff = timeDiffMinutes(groundTruth.sunsetUTC, calculated.sunsetUTC);
        
        const gtSunriseLocal = formatTime(groundTruth.sunriseUTC, tzOffset);
        const gtSunsetLocal = formatTime(groundTruth.sunsetUTC, tzOffset);
        const calcSunriseLocal = formatTime(calculated.sunriseUTC, tzOffset);
        const calcSunsetLocal = formatTime(calculated.sunsetUTC, tzOffset);
        
        const sunriseStatus = sunriseDiff <= TOLERANCE_MINUTES ? '✓ PASS' : '✗ FAIL';
        const sunsetStatus = sunsetDiff <= TOLERANCE_MINUTES ? '✓ PASS' : '✗ FAIL';
        
        console.log(`  Timezone: UTC${tzOffset >= 0 ? '+' : ''}${tzOffset}`);
        console.log(`  Sunrise:`);
        console.log(`    Ground Truth: ${gtSunriseLocal} (local)`);
        console.log(`    Calculated:   ${calcSunriseLocal} (local)`);
        console.log(`    Difference:   ${sunriseDiff.toFixed(1)} minutes ${sunriseStatus}`);
        console.log(`  Sunset:`);
        console.log(`    Ground Truth: ${gtSunsetLocal} (local)`);
        console.log(`    Calculated:   ${calcSunsetLocal} (local)`);
        console.log(`    Difference:   ${sunsetDiff.toFixed(1)} minutes ${sunsetStatus}`);
        
        totalTests += 2;
        if (sunriseDiff <= TOLERANCE_MINUTES) passCount++;
        else failCount++;
        if (sunsetDiff <= TOLERANCE_MINUTES) passCount++;
        else failCount++;
        
        await new Promise(r => setTimeout(r, 500));
    }
    
    console.log('\n========================================');
    console.log('SUMMARY');
    console.log('========================================');
    console.log(`✓ PASSED:  ${passCount}/${totalTests}`);
    console.log(`✗ FAILED:  ${failCount}/${totalTests}`);
    console.log(`Accuracy: ${((passCount/totalTests)*100).toFixed(1)}%`);
    console.log('========================================\n');
}

runTests().catch(console.error);
