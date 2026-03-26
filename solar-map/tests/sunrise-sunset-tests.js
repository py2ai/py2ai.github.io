const SunCalc = require('suncalc');
const https = require('https');

const TEST_DATE = '2025-03-26';
const TOLERANCE = {
    time: 30,
    altitude: 10,
    azimuth: 10,
    tilt: 2,
    daylight: 0.5
};

function getTimezoneOffset(lat, lon, date) {
    // Iran (UTC+3:30) - specific to Iran only
    if (lon >= 44 && lon <= 64 && lat >= 25 && lat <= 40) {
        // Exclude UAE and other Gulf states (they use UTC+4)
        if (lon >= 51 && lon <= 57 && lat >= 22 && lat <= 26) {
            return 4 * 60;
        }
        // Exclude Turkmenistan (uses UTC+5)
        if (lat >= 35 && lon >= 52 && lon <= 67) {
            return 5 * 60;
        }
        return 3.5 * 60;
    }
    if (lon >= 60 && lon <= 80 && lat >= 23 && lat <= 37) {
        return 5 * 60;
    }
    if (lon >= 68 && lon <= 97 && lat >= 6 && lat <= 35) {
        return 5.5 * 60;
    }
    if (lon >= 73 && lon <= 135 && lat >= 18 && lat <= 53) {
        if (lon >= 73 && lon <= 80 && lat >= 39 && lat <= 44) {
            return 6 * 60;
        }
        return 8 * 60;
    }
    if (lon >= 122 && lon <= 154 && lat >= 24 && lat <= 46) {
        return 9 * 60;
    }
    if (lat < -10 && lon > 110 && lon < 155) {
        if (lon > 140) {
            const month = date.getUTCMonth();
            const isDST = month < 4 || month > 9;
            return (isDST ? 11 : 10) * 60;
        }
        if (lon > 129 && lon <= 140) {
            const month = date.getUTCMonth();
            const isDST = month < 4 || month > 9;
            return (isDST ? 10.5 : 9.5) * 60;
        }
        return 8 * 60;
    }
    if (lat >= 24 && lat <= 50 && lon >= -125 && lon <= -66) {
        const month = date.getUTCMonth();
        const day = date.getUTCDate();
        const isDST = (month > 2 && month < 10) || (month === 2 && day >= 8) || (month === 10 && day < 7);
        if (lon >= -125 && lon < -110) return (isDST ? -7 : -8) * 60;
        if (lon >= -110 && lon < -100) return (isDST ? -6 : -7) * 60;
        if (lon >= -100 && lon < -85) return (isDST ? -5 : -6) * 60;
        return (isDST ? -4 : -5) * 60;
    }
    // Canada timezones
    if (lat >= 41 && lat <= 84 && lon >= -141 && lon <= -52) {
        const month = date.getUTCMonth();
        const day = date.getUTCDate();
        const isDST = (month > 2 && month < 10) || (month === 2 && day >= 8) || (month === 10 && day < 7);
        if (lon >= -141 && lon < -125) return (isDST ? -7 : -8) * 60;
        if (lon >= -125 && lon < -102) return (isDST ? -6 : -7) * 60;
        if (lon >= -102 && lon < -85) return (isDST ? -5 : -6) * 60;
        if (lon >= -85 && lon < -66) return (isDST ? -4 : -5) * 60;
        return (isDST ? -3 : -4) * 60;
    }
    if (lat >= 35 && lat <= 70 && lon >= -10 && lon <= 40) {
        const month = date.getUTCMonth();
        const isDST = month > 2 && month < 10;
        const baseOffset = Math.round(lon / 15) * 60;
        return baseOffset + (isDST ? 60 : 0);
    }
    const lonOffset = Math.round(lon / 15);
    const baseOffset = lonOffset * 60;
    const month = date.getUTCMonth();
    const day = date.getUTCDate();
    let isDST = false;
    if (lat > 23.5) {
        isDST = (month > 2 && month < 10) || (month === 2 && day >= 8) || (month === 10 && day < 7);
    } else if (lat < -23.5) {
        isDST = month < 4 || month > 9;
    }
    return baseOffset + (isDST ? 60 : 0);
}

function getSunriseSunsetLocal(lat, lon, dateStr) {
    const [year, month, day] = dateStr.split('-').map(Number);
    const date = new Date(Date.UTC(year, month - 1, day, 12, 0, 0));
    const times = SunCalc.getTimes(date, lat, lon);
    
    const tzOffset = getTimezoneOffset(lat, lon, date);
    
    const sunriseUTC = times.sunrise;
    const sunsetUTC = times.sunset;
    
    const sunriseLocalMinutes = sunriseUTC.getUTCHours() * 60 + sunriseUTC.getUTCMinutes() + tzOffset;
    const sunsetLocalMinutes = sunsetUTC.getUTCHours() * 60 + sunsetUTC.getUTCMinutes() + tzOffset;
    
    let sunriseHours = Math.floor(sunriseLocalMinutes / 60);
    let sunriseMins = sunriseLocalMinutes % 60;
    while (sunriseHours < 0) sunriseHours += 24;
    while (sunriseHours >= 24) sunriseHours -= 24;
    while (sunriseMins < 0) sunriseMins += 60;
    while (sunriseMins >= 60) sunriseMins -= 60;
    
    let sunsetHours = Math.floor(sunsetLocalMinutes / 60);
    let sunsetMins = sunsetLocalMinutes % 60;
    while (sunsetHours < 0) sunsetHours += 24;
    while (sunsetHours >= 24) sunsetHours -= 24;
    while (sunsetMins < 0) sunsetMins += 60;
    while (sunsetMins >= 60) sunsetMins -= 60;
    
    return {
        sunrise: { hour: sunriseHours, minute: Math.round(sunriseMins) },
        sunset: { hour: sunsetHours, minute: Math.round(sunsetMins) },
        daylight: (times.sunset - times.sunrise) / (1000 * 60 * 60),
        sunriseUTC: times.sunrise,
        sunsetUTC: times.sunset,
        tzOffset: tzOffset
    };
}

const testLocations = [
    { name: 'Afghanistan', lat: 34.5553, lon: 69.2075 },
    { name: 'Albania', lat: 41.1533, lon: 20.1683 },
    { name: 'Algeria', lat: 28.0339, lon: 1.6596 },
    { name: 'Andorra', lat: 42.5063, lon: 1.5218 },
    { name: 'Angola', lat: -11.2027, lon: 17.8739 },
    { name: 'Antigua and Barbuda', lat: 17.0608, lon: -61.7964 },
    { name: 'Argentina', lat: -38.4161, lon: -63.6167 },
    { name: 'Armenia', lat: 40.0691, lon: 45.0382 },
    { name: 'Australia', lat: -25.2744, lon: 133.7751 },
    { name: 'Austria', lat: 47.5162, lon: 14.5501 },
    { name: 'Azerbaijan', lat: 40.1431, lon: 47.5769 },
    { name: 'Bahamas', lat: 25.0343, lon: -77.3963 },
    { name: 'Bahrain', lat: 25.9304, lon: 50.6378 },
    { name: 'Bangladesh', lat: 23.6850, lon: 90.3563 },
    { name: 'Barbados', lat: 13.1939, lon: -59.5432 },
    { name: 'Belarus', lat: 53.7098, lon: 27.9534 },
    { name: 'Belgium', lat: 50.5039, lon: 4.4699 },
    { name: 'Belize', lat: 17.1899, lon: -88.4976 },
    { name: 'Benin', lat: 9.3077, lon: 2.3158 },
    { name: 'Bhutan', lat: 27.5142, lon: 90.4336 },
    { name: 'Bolivia', lat: -16.2902, lon: -63.5887 },
    { name: 'Bosnia and Herzegovina', lat: 43.9159, lon: 17.6791 },
    { name: 'Botswana', lat: -22.3285, lon: 24.6849 },
    { name: 'Brazil', lat: -14.2350, lon: -51.9253 },
    { name: 'Brunei', lat: 4.5353, lon: 114.7277 },
    { name: 'Bulgaria', lat: 42.7339, lon: 25.4858 },
    { name: 'Burkina Faso', lat: 12.2383, lon: -1.5616 },
    { name: 'Burundi', lat: -3.3731, lon: 29.9189 },
    { name: 'Cambodia', lat: 12.5657, lon: 104.9910 },
    { name: 'Cameroon', lat: 7.3697, lon: 12.3547 },
    { name: 'Canada', lat: 56.1304, lon: -106.3468 },
    { name: 'Cape Verde', lat: 16.5388, lon: -23.0418 },
    { name: 'Central African Republic', lat: 6.6111, lon: 20.9394 },
    { name: 'Chad', lat: 15.4542, lon: 18.7322 },
    { name: 'Chile', lat: -35.6751, lon: -71.5430 },
    { name: 'China', lat: 35.8617, lon: 104.1954 },
    { name: 'Colombia', lat: 4.5709, lon: -74.2973 },
    { name: 'Comoros', lat: -11.6455, lon: 43.3333 },
    { name: 'Congo', lat: -0.2280, lon: 15.8277 },
    { name: 'Costa Rica', lat: 9.7489, lon: -83.7534 },
    { name: 'Croatia', lat: 45.1000, lon: 15.2000 },
    { name: 'Cuba', lat: 21.5218, lon: -77.7812 },
    { name: 'Cyprus', lat: 35.1264, lon: 33.4299 },
    { name: 'Czech Republic', lat: 49.8175, lon: 15.4730 },
    { name: 'Denmark', lat: 56.2639, lon: 9.5018 },
    { name: 'Djibouti', lat: 11.8251, lon: 42.5903 },
    { name: 'Dominica', lat: 15.4150, lon: -61.3710 },
    { name: 'Dominican Republic', lat: 18.7357, lon: -70.1627 },
    { name: 'East Timor', lat: -8.8742, lon: 125.7275 },
    { name: 'Ecuador', lat: -1.8312, lon: -78.1834 },
    { name: 'Egypt', lat: 26.8206, lon: 30.8025 },
    { name: 'El Salvador', lat: 13.7942, lon: -88.8965 },
    { name: 'Equatorial Guinea', lat: 1.6508, lon: 10.2679 },
    { name: 'Eritrea', lat: 15.1794, lon: 39.7823 },
    { name: 'Estonia', lat: 58.5953, lon: 25.0136 },
    { name: 'Eswatini', lat: -26.5225, lon: 31.4659 },
    { name: 'Ethiopia', lat: 9.1450, lon: 40.4897 },
    { name: 'Fiji', lat: -17.7134, lon: 178.0650 },
    { name: 'Finland', lat: 61.9241, lon: 25.7482 },
    { name: 'France', lat: 46.2276, lon: 2.2137 },
    { name: 'Gabon', lat: -0.8037, lon: 11.6094 },
    { name: 'Gambia', lat: 13.4432, lon: -15.3101 },
    { name: 'Georgia', lat: 42.3154, lon: 43.3569 },
    { name: 'Germany', lat: 51.1657, lon: 10.4515 },
    { name: 'Ghana', lat: 7.9465, lon: -1.0232 },
    { name: 'Greece', lat: 39.0742, lon: 21.8243 },
    { name: 'Grenada', lat: 12.1165, lon: -61.6790 },
    { name: 'Guatemala', lat: 15.7835, lon: -90.2308 },
    { name: 'Guinea', lat: 9.9456, lon: -9.6966 },
    { name: 'Guinea-Bissau', lat: 11.8037, lon: -15.1804 },
    { name: 'Guyana', lat: 4.8604, lon: -58.9302 },
    { name: 'Haiti', lat: 18.9712, lon: -72.2852 },
    { name: 'Honduras', lat: 15.2000, lon: -86.2419 },
    { name: 'Hungary', lat: 47.1625, lon: 19.5033 },
    { name: 'Iceland', lat: 64.9631, lon: -19.0208 },
    { name: 'India', lat: 20.5937, lon: 78.9629 },
    { name: 'Indonesia', lat: -0.7893, lon: 113.9213 },
    { name: 'Iran', lat: 32.4279, lon: 53.6880 },
    { name: 'Iraq', lat: 33.2232, lon: 43.6793 },
    { name: 'Ireland', lat: 53.1424, lon: -7.6921 },
    { name: 'Israel', lat: 31.0461, lon: 34.8516 },
    { name: 'Italy', lat: 41.8719, lon: 12.5674 },
    { name: 'Ivory Coast', lat: 7.5400, lon: -5.5471 },
    { name: 'Jamaica', lat: 18.1096, lon: -77.2975 },
    { name: 'Japan', lat: 36.2048, lon: 138.2529 },
    { name: 'Jordan', lat: 30.5852, lon: 36.2384 },
    { name: 'Kazakhstan', lat: 48.0196, lon: 66.9237 },
    { name: 'Kenya', lat: -0.0236, lon: 37.9062 },
    { name: 'Kiribati', lat: -3.3704, lon: -168.7340 },
    { name: 'Kosovo', lat: 42.6026, lon: 20.9030 },
    { name: 'Kuwait', lat: 29.3117, lon: 47.4818 },
    { name: 'Kyrgyzstan', lat: 41.2044, lon: 75.7500 },
    { name: 'Laos', lat: 19.8563, lon: 102.4955 },
    { name: 'Latvia', lat: 56.8796, lon: 24.6032 },
    { name: 'Lebanon', lat: 33.8547, lon: 35.8623 },
    { name: 'Lesotho', lat: -29.6100, lon: 28.2336 },
    { name: 'Liberia', lat: 6.4281, lon: -9.4295 },
    { name: 'Libya', lat: 26.3351, lon: 16.2288 },
    { name: 'Liechtenstein', lat: 47.1660, lon: 9.5554 },
    { name: 'Lithuania', lat: 55.1694, lon: 26.0164 },
    { name: 'Luxembourg', lat: 49.8153, lon: 6.1296 },
    { name: 'Madagascar', lat: -18.7669, lon: 46.8691 },
    { name: 'Malawi', lat: -13.2543, lon: 34.3013 },
    { name: 'Malaysia', lat: 4.2105, lon: 101.9758 },
    { name: 'Maldives', lat: 3.2028, lon: 73.2207 },
    { name: 'Mali', lat: 17.5707, lon: -3.9962 },
    { name: 'Malta', lat: 35.9375, lon: 14.3754 },
    { name: 'Marshall Islands', lat: 7.1315, lon: 171.1845 },
    { name: 'Mauritania', lat: 21.0079, lon: -10.9408 },
    { name: 'Mauritius', lat: -20.3484, lon: 57.5522 },
    { name: 'Mexico', lat: 23.6345, lon: -102.5528 },
    { name: 'Micronesia', lat: 7.4256, lon: 150.5506 },
    { name: 'Moldova', lat: 47.4116, lon: 30.0643 },
    { name: 'Monaco', lat: 43.7384, lon: 7.4246 },
    { name: 'Mongolia', lat: 46.8625, lon: 92.2372 },
    { name: 'Montenegro', lat: 42.7087, lon: 19.3744 },
    { name: 'Morocco', lat: 31.7917, lon: -7.0926 },
    { name: 'Mozambique', lat: -18.6657, lon: 35.5296 },
    { name: 'Myanmar', lat: 21.9162, lon: 95.9560 },
    { name: 'Namibia', lat: -22.9576, lon: 17.0832 },
    { name: 'Nauru', lat: -0.5228, lon: 166.9315 },
    { name: 'Nepal', lat: 28.3949, lon: 84.1240 },
    { name: 'Netherlands', lat: 52.1326, lon: 5.2913 },
    { name: 'New Zealand', lat: -40.9006, lon: 174.8860 },
    { name: 'Nicaragua', lat: 12.8654, lon: -86.0458 },
    { name: 'Niger', lat: 17.6078, lon: -10.4311 },
    { name: 'Nigeria', lat: 9.0820, lon: 8.6753 },
    { name: 'North Korea', lat: 40.3399, lon: 127.5101 },
    { name: 'North Macedonia', lat: 41.5124, lon: 22.9957 },
    { name: 'Norway', lat: 60.4720, lon: 8.4689 },
    { name: 'Oman', lat: 21.4735, lon: 55.9804 },
    { name: 'Pakistan', lat: 30.3753, lon: 69.3451 },
    { name: 'Palau', lat: 7.5150, lon: 134.5825 },
    { name: 'Palestine', lat: 31.9522, lon: 35.2332 },
    { name: 'Panama', lat: 8.5380, lon: -80.7821 },
    { name: 'Papua New Guinea', lat: -6.3150, lon: 143.9555 },
    { name: 'Paraguay', lat: -23.4425, lon: -58.4438 },
    { name: 'Peru', lat: -9.1900, lon: -75.0152 },
    { name: 'Philippines', lat: 12.8797, lon: 121.7740 },
    { name: 'Poland', lat: 51.9194, lon: 19.1451 },
    { name: 'Portugal', lat: 39.3999, lon: -8.2245 },
    { name: 'Qatar', lat: 25.3548, lon: 51.1839 },
    { name: 'Romania', lat: 45.9432, lon: 24.9668 },
    { name: 'Russia', lat: 61.5240, lon: 105.3188 },
    { name: 'Rwanda', lat: -1.9403, lon: 29.8739 },
    { name: 'Saint Kitts and Nevis', lat: 17.3578, lon: -62.7830 },
    { name: 'Saint Lucia', lat: 13.9094, lon: -60.9789 },
    { name: 'Saint Vincent and the Grenadines', lat: 12.9843, lon: -61.2872 },
    { name: 'Samoa', lat: -13.7590, lon: -172.1046 },
    { name: 'San Marino', lat: 43.9424, lon: 12.4578 },
    { name: 'Sao Tome and Principe', lat: 0.1864, lon: 6.6131 },
    { name: 'Saudi Arabia', lat: 23.8859, lon: 45.0792 },
    { name: 'Senegal', lat: 14.4974, lon: -14.4524 },
    { name: 'Serbia', lat: 44.0165, lon: 20.5081 },
    { name: 'Seychelles', lat: -4.6796, lon: 55.4920 },
    { name: 'Sierra Leone', lat: 8.4606, lon: -11.7799 },
    { name: 'Singapore', lat: 1.3521, lon: 103.8198 },
    { name: 'Slovakia', lat: 48.6690, lon: 19.6990 },
    { name: 'Slovenia', lat: 46.1512, lon: 14.9955 },
    { name: 'Solomon Islands', lat: -9.6457, lon: 160.1562 },
    { name: 'Somalia', lat: 5.1521, lon: 46.1996 },
    { name: 'South Africa', lat: -30.5595, lon: 22.9375 },
    { name: 'South Korea', lat: 35.9078, lon: 127.7669 },
    { name: 'South Sudan', lat: 6.8770, lon: 31.3070 },
    { name: 'Spain', lat: 40.4637, lon: -3.7492 },
    { name: 'Sri Lanka', lat: 7.8731, lon: 80.7718 },
    { name: 'Sudan', lat: 12.8628, lon: 30.2176 },
    { name: 'Suriname', lat: 3.9193, lon: -56.0278 },
    { name: 'Sweden', lat: 60.1282, lon: 18.6435 },
    { name: 'Switzerland', lat: 46.8182, lon: 8.2275 },
    { name: 'Syria', lat: 34.8021, lon: 39.0000 },
    { name: 'Taiwan', lat: 23.6978, lon: 120.9605 },
    { name: 'Tajikistan', lat: 38.8610, lon: 71.2761 },
    { name: 'Tanzania', lat: -6.3690, lon: 34.8888 },
    { name: 'Thailand', lat: 15.8700, lon: 100.9925 },
    { name: 'Togo', lat: 8.6195, lon: 0.8248 },
    { name: 'Tonga', lat: -21.1790, lon: -175.1982 },
    { name: 'Trinidad and Tobago', lat: 10.6918, lon: -61.2225 },
    { name: 'Tunisia', lat: 33.8869, lon: 9.5375 },
    { name: 'Turkey', lat: 38.9637, lon: 35.2433 },
    { name: 'Turkmenistan', lat: 38.9697, lon: 59.5563 },
    { name: 'Tuvalu', lat: -7.1095, lon: 177.6493 },
    { name: 'Uganda', lat: 1.3733, lon: 32.2903 },
    { name: 'Ukraine', lat: 48.3794, lon: 31.1656 },
    { name: 'United Arab Emirates', lat: 23.4241, lon: 53.8475 },
    { name: 'United Kingdom', lat: 55.3781, lon: -3.4360 },
    { name: 'United States', lat: 37.0902, lon: -95.7129 },
    { name: 'Uruguay', lat: -32.5228, lon: -55.7658 },
    { name: 'Uzbekistan', lat: 41.3775, lon: 64.5853 },
    { name: 'Vanuatu', lat: -15.3767, lon: 166.9592 },
    { name: 'Vatican City', lat: 41.9029, lon: 12.4534 },
    { name: 'Venezuela', lat: 6.4238, lon: -66.5897 },
    { name: 'Vietnam', lat: 14.0583, lon: 108.2772 },
    { name: 'Yemen', lat: 15.5527, lon: 48.5164 },
    { name: 'Zambia', lat: -13.1339, lon: 27.8493 },
    { name: 'Zimbabwe', lat: -19.0154, lon: 29.1549 },
    { name: 'American Samoa', lat: -14.2708, lon: -170.7020 },
    { name: 'Anguilla', lat: 18.2206, lon: -63.0686 },
    { name: 'Aruba', lat: 12.5211, lon: -69.9683 },
    { name: 'Bermuda', lat: 32.3078, lon: -64.7505 },
    { name: 'British Virgin Islands', lat: 18.4207, lon: -64.6399 },
    { name: 'Cayman Islands', lat: 19.3133, lon: -81.2546 },
    { name: 'Christmas Island', lat: -10.4475, lon: 105.6904 },
    { name: 'Cocos Islands', lat: -12.1642, lon: 96.8710 },
    { name: 'Cook Islands', lat: -21.2367, lon: -159.7777 },
    { name: 'Curacao', lat: 12.1696, lon: -68.9900 },
    { name: 'Falkland Islands', lat: -51.7963, lon: -59.5236 },
    { name: 'Faroe Islands', lat: 61.8926, lon: -6.9118 },
    { name: 'French Polynesia', lat: -17.6797, lon: -149.4068 },
    { name: 'Gibraltar', lat: 36.1408, lon: -5.3536 },
    { name: 'Greenland', lat: 71.7069, lon: -42.6043 },
    { name: 'Guam', lat: 13.4443, lon: 144.7937 },
    { name: 'Guernsey', lat: 49.4657, lon: -2.5853 },
    { name: 'Hong Kong', lat: 22.3193, lon: 114.1694 },
    { name: 'Jersey', lat: 49.2144, lon: -2.1313 },
    { name: 'Macau', lat: 22.1987, lon: 113.5439 },
    { name: 'Mayotte', lat: -12.8275, lon: 45.1662 },
    { name: 'Montserrat', lat: 16.7425, lon: -62.1874 },
    { name: 'New Caledonia', lat: -20.9043, lon: 165.6180 },
    { name: 'Niue', lat: -19.0544, lon: -169.8672 },
    { name: 'Norfolk Island', lat: -29.0408, lon: 167.9547 },
    { name: 'Northern Mariana Islands', lat: 17.3302, lon: 145.3845 },
    { name: 'Pitcairn Islands', lat: -24.7036, lon: -127.4393 },
    { name: 'Puerto Rico', lat: 18.2208, lon: -66.5901 },
    { name: 'Reunion', lat: -21.1151, lon: 55.5364 },
    { name: 'Sint Maarten', lat: 18.0425, lon: -63.0548 },
    { name: 'Svalbard', lat: 77.5536, lon: 23.6703 },
    { name: 'Tokelau', lat: -8.9674, lon: -171.8559 },
    { name: 'Turks and Caicos Islands', lat: 21.6940, lon: -71.7979 },
    { name: 'US Virgin Islands', lat: 18.3358, lon: -64.8963 },
    { name: 'Wallis and Futuna', lat: -13.7687, lon: -177.1561 }
];

let passCount = 0;
let failCount = 0;
let warningCount = 0;

console.log('\n========================================');
console.log('SUNRISE/SUNSET TIME TESTS - ALL 232 COUNTRIES');
console.log('Date: ' + TEST_DATE + ' (near equinox)');
console.log('Times shown are LOCAL CLOCK TIME');
console.log('========================================\n');

console.log('TEST: Sunrise/Sunset Local Times with Timezone');
console.log('------------------------------------------------');
testLocations.forEach(loc => {
    const times = getSunriseSunsetLocal(loc.lat, loc.lon, TEST_DATE);
    const sunriseStr = `${times.sunrise.hour.toString().padStart(2, '0')}:${times.sunrise.minute.toString().padStart(2, '0')}`;
    const sunsetStr = `${times.sunset.hour.toString().padStart(2, '0')}:${times.sunset.minute.toString().padStart(2, '0')}`;
    const tzHours = times.tzOffset / 60;
    const tzStr = tzHours >= 0 ? `UTC+${tzHours}` : `UTC${tzHours}`;
    
    const sunriseOk = times.sunrise.hour >= 5 && times.sunrise.hour <= 8;
    const sunsetOk = times.sunset.hour >= 17 && times.sunset.hour <= 20;
    
    let status;
    if (sunriseOk && sunsetOk) {
        status = '✓ PASS';
        passCount++;
    } else if (times.sunrise.hour >= 4 && times.sunrise.hour <= 9 && times.sunset.hour >= 16 && times.sunset.hour <= 21) {
        status = '⚠ WARN (edge case)';
        warningCount++;
    } else {
        status = '✗ FAIL (unusual times)';
        failCount++;
    }
    
    console.log(`${loc.name}: Sunrise=${sunriseStr} Sunset=${sunsetStr} TZ=${tzStr} Daylight=${times.daylight.toFixed(1)}h ${status}`);
});

console.log('\n========================================');
console.log('SUMMARY');
console.log('========================================');
console.log(`✓ PASSED:  ${passCount}`);
console.log(`⚠ WARNINGS: ${warningCount}`);
console.log(`✗ FAILED:  ${failCount}`);
console.log(`TOTAL:     ${passCount + warningCount + failCount}`);
console.log(`Countries: ${testLocations.length}`);
console.log('========================================\n');
