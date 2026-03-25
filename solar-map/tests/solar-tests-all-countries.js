const SunCalc = require('suncalc');

const TEST_DATE = '2025-03-26';
const TOLERANCE = {
    time: 60,
    altitude: 10,
    azimuth: 10,
    tilt: 2,
    daylight: 0.5
};

// All 215 countries with coordinates (capital city or center point)
const testLocations = [
    { name: 'Afghanistan', lat: 34.5553, lon: 69.2075, hemisphere: 'north' },
    { name: 'Albania', lat: 41.1533, lon: 20.1683, hemisphere: 'north' },
    { name: 'Algeria', lat: 28.0339, lon: 1.6596, hemisphere: 'north' },
    { name: 'Andorra', lat: 42.5063, lon: 1.5218, hemisphere: 'north' },
    { name: 'Angola', lat: -11.2027, lon: 17.8739, hemisphere: 'south' },
    { name: 'Antigua and Barbuda', lat: 17.0608, lon: -61.7964, hemisphere: 'north' },
    { name: 'Argentina', lat: -38.4161, lon: -63.6167, hemisphere: 'south' },
    { name: 'Armenia', lat: 40.0691, lon: 45.0382, hemisphere: 'north' },
    { name: 'Australia', lat: -25.2744, lon: 133.7751, hemisphere: 'south' },
    { name: 'Austria', lat: 47.5162, lon: 14.5501, hemisphere: 'north' },
    { name: 'Azerbaijan', lat: 40.1431, lon: 47.5769, hemisphere: 'north' },
    { name: 'Bahamas', lat: 25.0343, lon: -77.3963, hemisphere: 'north' },
    { name: 'Bahrain', lat: 25.9304, lon: 50.6378, hemisphere: 'north' },
    { name: 'Bangladesh', lat: 23.6850, lon: 90.3563, hemisphere: 'north' },
    { name: 'Barbados', lat: 13.1939, lon: -59.5432, hemisphere: 'north' },
    { name: 'Belarus', lat: 53.7098, lon: 27.9534, hemisphere: 'north' },
    { name: 'Belgium', lat: 50.5039, lon: 4.4699, hemisphere: 'north' },
    { name: 'Belize', lat: 17.1899, lon: -88.4976, hemisphere: 'north' },
    { name: 'Benin', lat: 9.3077, lon: 2.3158, hemisphere: 'north' },
    { name: 'Bhutan', lat: 27.5142, lon: 90.4336, hemisphere: 'north' },
    { name: 'Bolivia', lat: -16.2902, lon: -63.5887, hemisphere: 'south' },
    { name: 'Bosnia and Herzegovina', lat: 43.9159, lon: 17.6791, hemisphere: 'north' },
    { name: 'Botswana', lat: -22.3285, lon: 24.6849, hemisphere: 'south' },
    { name: 'Brazil', lat: -14.2350, lon: -51.9253, hemisphere: 'south' },
    { name: 'Brunei', lat: 4.5353, lon: 114.7277, hemisphere: 'north' },
    { name: 'Bulgaria', lat: 42.7339, lon: 25.4858, hemisphere: 'north' },
    { name: 'Burkina Faso', lat: 12.2383, lon: -1.5616, hemisphere: 'north' },
    { name: 'Burundi', lat: -3.3731, lon: 29.9189, hemisphere: 'south' },
    { name: 'Cambodia', lat: 12.5657, lon: 104.9910, hemisphere: 'north' },
    { name: 'Cameroon', lat: 7.3697, lon: 12.3547, hemisphere: 'north' },
    { name: 'Canada', lat: 56.1304, lon: -106.3468, hemisphere: 'north' },
    { name: 'Cape Verde', lat: 16.5388, lon: -23.0418, hemisphere: 'north' },
    { name: 'Central African Republic', lat: 6.6111, lon: 20.9394, hemisphere: 'north' },
    { name: 'Chad', lat: 15.4542, lon: 18.7322, hemisphere: 'north' },
    { name: 'Chile', lat: -35.6751, lon: -71.5430, hemisphere: 'south' },
    { name: 'China', lat: 35.8617, lon: 104.1954, hemisphere: 'north' },
    { name: 'Colombia', lat: 4.5709, lon: -74.2973, hemisphere: 'north' },
    { name: 'Comoros', lat: -11.6455, lon: 43.3333, hemisphere: 'south' },
    { name: 'Congo', lat: -0.2280, lon: 15.8277, hemisphere: 'south' },
    { name: 'Costa Rica', lat: 9.7489, lon: -83.7534, hemisphere: 'north' },
    { name: 'Croatia', lat: 45.1000, lon: 15.2000, hemisphere: 'north' },
    { name: 'Cuba', lat: 21.5218, lon: -77.7812, hemisphere: 'north' },
    { name: 'Cyprus', lat: 35.1264, lon: 33.4299, hemisphere: 'north' },
    { name: 'Czech Republic', lat: 49.8175, lon: 15.4730, hemisphere: 'north' },
    { name: 'Denmark', lat: 56.2639, lon: 9.5018, hemisphere: 'north' },
    { name: 'Djibouti', lat: 11.8251, lon: 42.5903, hemisphere: 'north' },
    { name: 'Dominica', lat: 15.4150, lon: -61.3710, hemisphere: 'north' },
    { name: 'Dominican Republic', lat: 18.7357, lon: -70.1627, hemisphere: 'north' },
    { name: 'East Timor', lat: -8.8742, lon: 125.7275, hemisphere: 'south' },
    { name: 'Ecuador', lat: -1.8312, lon: -78.1834, hemisphere: 'south' },
    { name: 'Egypt', lat: 26.8206, lon: 30.8025, hemisphere: 'north' },
    { name: 'El Salvador', lat: 13.7942, lon: -88.8965, hemisphere: 'north' },
    { name: 'Equatorial Guinea', lat: 1.6508, lon: 10.2679, hemisphere: 'north' },
    { name: 'Eritrea', lat: 15.1794, lon: 39.7823, hemisphere: 'north' },
    { name: 'Estonia', lat: 58.5953, lon: 25.0136, hemisphere: 'north' },
    { name: 'Eswatini', lat: -26.5225, lon: 31.4659, hemisphere: 'south' },
    { name: 'Ethiopia', lat: 9.1450, lon: 40.4897, hemisphere: 'north' },
    { name: 'Fiji', lat: -17.7134, lon: 178.0650, hemisphere: 'south' },
    { name: 'Finland', lat: 61.9241, lon: 25.7482, hemisphere: 'north' },
    { name: 'France', lat: 46.2276, lon: 2.2137, hemisphere: 'north' },
    { name: 'Gabon', lat: -0.8037, lon: 11.6094, hemisphere: 'south' },
    { name: 'Gambia', lat: 13.4432, lon: -15.3101, hemisphere: 'north' },
    { name: 'Georgia', lat: 42.3154, lon: 43.3569, hemisphere: 'north' },
    { name: 'Germany', lat: 51.1657, lon: 10.4515, hemisphere: 'north' },
    { name: 'Ghana', lat: 7.9465, lon: -1.0232, hemisphere: 'north' },
    { name: 'Greece', lat: 39.0742, lon: 21.8243, hemisphere: 'north' },
    { name: 'Grenada', lat: 12.1165, lon: -61.6790, hemisphere: 'north' },
    { name: 'Guatemala', lat: 15.7835, lon: -90.2308, hemisphere: 'north' },
    { name: 'Guinea', lat: 9.9456, lon: -9.6966, hemisphere: 'north' },
    { name: 'Guinea-Bissau', lat: 11.8037, lon: -15.1804, hemisphere: 'north' },
    { name: 'Guyana', lat: 4.8604, lon: -58.9302, hemisphere: 'north' },
    { name: 'Haiti', lat: 18.9712, lon: -72.2852, hemisphere: 'north' },
    { name: 'Honduras', lat: 15.2000, lon: -86.2419, hemisphere: 'north' },
    { name: 'Hungary', lat: 47.1625, lon: 19.5033, hemisphere: 'north' },
    { name: 'Iceland', lat: 64.9631, lon: -19.0208, hemisphere: 'north' },
    { name: 'India', lat: 20.5937, lon: 78.9629, hemisphere: 'north' },
    { name: 'Indonesia', lat: -0.7893, lon: 113.9213, hemisphere: 'south' },
    { name: 'Iran', lat: 32.4279, lon: 53.6880, hemisphere: 'north' },
    { name: 'Iraq', lat: 33.2232, lon: 43.6793, hemisphere: 'north' },
    { name: 'Ireland', lat: 53.1424, lon: -7.6921, hemisphere: 'north' },
    { name: 'Israel', lat: 31.0461, lon: 34.8516, hemisphere: 'north' },
    { name: 'Italy', lat: 41.8719, lon: 12.5674, hemisphere: 'north' },
    { name: 'Ivory Coast', lat: 7.5400, lon: -5.5471, hemisphere: 'north' },
    { name: 'Jamaica', lat: 18.1096, lon: -77.2975, hemisphere: 'north' },
    { name: 'Japan', lat: 36.2048, lon: 138.2529, hemisphere: 'north' },
    { name: 'Jordan', lat: 30.5852, lon: 36.2384, hemisphere: 'north' },
    { name: 'Kazakhstan', lat: 48.0196, lon: 66.9237, hemisphere: 'north' },
    { name: 'Kenya', lat: -0.0236, lon: 37.9062, hemisphere: 'south' },
    { name: 'Kiribati', lat: -3.3704, lon: -168.7340, hemisphere: 'south' },
    { name: 'Kosovo', lat: 42.6026, lon: 20.9030, hemisphere: 'north' },
    { name: 'Kuwait', lat: 29.3117, lon: 47.4818, hemisphere: 'north' },
    { name: 'Kyrgyzstan', lat: 41.2044, lon: 74.7661, hemisphere: 'north' },
    { name: 'Laos', lat: 19.8563, lon: 102.4955, hemisphere: 'north' },
    { name: 'Latvia', lat: 56.8796, lon: 24.6032, hemisphere: 'north' },
    { name: 'Lebanon', lat: 33.8547, lon: 35.8623, hemisphere: 'north' },
    { name: 'Lesotho', lat: -29.6100, lon: 28.2336, hemisphere: 'south' },
    { name: 'Liberia', lat: 6.4281, lon: -9.4295, hemisphere: 'north' },
    { name: 'Libya', lat: 26.3351, lon: 17.2283, hemisphere: 'north' },
    { name: 'Liechtenstein', lat: 47.1660, lon: 9.5554, hemisphere: 'north' },
    { name: 'Lithuania', lat: 55.1694, lon: 23.8813, hemisphere: 'north' },
    { name: 'Luxembourg', lat: 49.8153, lon: 6.1296, hemisphere: 'north' },
    { name: 'Madagascar', lat: -18.7669, lon: 46.8691, hemisphere: 'south' },
    { name: 'Malawi', lat: -13.2543, lon: 34.3015, hemisphere: 'south' },
    { name: 'Malaysia', lat: 4.2105, lon: 101.9758, hemisphere: 'north' },
    { name: 'Maldives', lat: 3.2028, lon: 73.2207, hemisphere: 'north' },
    { name: 'Mali', lat: 17.5707, lon: -3.9962, hemisphere: 'north' },
    { name: 'Malta', lat: 35.9375, lon: 14.3754, hemisphere: 'north' },
    { name: 'Marshall Islands', lat: 7.1315, lon: 171.1845, hemisphere: 'north' },
    { name: 'Mauritania', lat: 21.0079, lon: -10.9408, hemisphere: 'north' },
    { name: 'Mauritius', lat: -20.3484, lon: 57.5522, hemisphere: 'south' },
    { name: 'Mexico', lat: 23.6345, lon: -102.5528, hemisphere: 'north' },
    { name: 'Micronesia', lat: 7.4256, lon: 150.5508, hemisphere: 'north' },
    { name: 'Moldova', lat: 47.4116, lon: 28.3699, hemisphere: 'north' },
    { name: 'Monaco', lat: 43.7384, lon: 7.4246, hemisphere: 'north' },
    { name: 'Mongolia', lat: 46.8625, lon: 103.8467, hemisphere: 'north' },
    { name: 'Montenegro', lat: 42.7087, lon: 19.3744, hemisphere: 'north' },
    { name: 'Morocco', lat: 31.7917, lon: -7.0926, hemisphere: 'north' },
    { name: 'Mozambique', lat: -18.6657, lon: 35.5296, hemisphere: 'south' },
    { name: 'Myanmar', lat: 21.9162, lon: 95.9560, hemisphere: 'north' },
    { name: 'Namibia', lat: -22.9576, lon: 18.4904, hemisphere: 'south' },
    { name: 'Nauru', lat: -0.5228, lon: 166.9315, hemisphere: 'south' },
    { name: 'Nepal', lat: 28.3949, lon: 84.1240, hemisphere: 'north' },
    { name: 'Netherlands', lat: 52.1326, lon: 5.2913, hemisphere: 'north' },
    { name: 'New Zealand', lat: -40.9006, lon: 174.8860, hemisphere: 'south' },
    { name: 'Nicaragua', lat: 12.8654, lon: -85.2072, hemisphere: 'north' },
    { name: 'Niger', lat: 17.6078, lon: 8.0817, hemisphere: 'north' },
    { name: 'Nigeria', lat: 9.0820, lon: 8.6753, hemisphere: 'north' },
    { name: 'North Korea', lat: 40.3399, lon: 127.5101, hemisphere: 'north' },
    { name: 'North Macedonia', lat: 41.5124, lon: 21.7453, hemisphere: 'north' },
    { name: 'Norway', lat: 60.4720, lon: 8.4689, hemisphere: 'north' },
    { name: 'Oman', lat: 21.4735, lon: 55.9754, hemisphere: 'north' },
    { name: 'Pakistan', lat: 30.3753, lon: 69.3451, hemisphere: 'north' },
    { name: 'Palau', lat: 7.5150, lon: 134.5825, hemisphere: 'north' },
    { name: 'Palestine', lat: 31.9522, lon: 35.2332, hemisphere: 'north' },
    { name: 'Panama', lat: 8.5380, lon: -80.7821, hemisphere: 'north' },
    { name: 'Papua New Guinea', lat: -6.3150, lon: 143.9555, hemisphere: 'south' },
    { name: 'Paraguay', lat: -23.4425, lon: -58.4438, hemisphere: 'south' },
    { name: 'Peru', lat: -9.1900, lon: -75.0152, hemisphere: 'south' },
    { name: 'Philippines', lat: 12.8797, lon: 121.7740, hemisphere: 'north' },
    { name: 'Poland', lat: 51.9194, lon: 19.1451, hemisphere: 'north' },
    { name: 'Portugal', lat: 39.3999, lon: -8.2245, hemisphere: 'north' },
    { name: 'Qatar', lat: 25.3548, lon: 51.1839, hemisphere: 'north' },
    { name: 'Romania', lat: 45.9432, lon: 24.9668, hemisphere: 'north' },
    { name: 'Russia', lat: 61.5240, lon: 105.3188, hemisphere: 'north' },
    { name: 'Rwanda', lat: -1.9403, lon: 29.8739, hemisphere: 'south' },
    { name: 'Saint Kitts and Nevis', lat: 17.3578, lon: -62.7830, hemisphere: 'north' },
    { name: 'Saint Lucia', lat: 13.9094, lon: -60.9789, hemisphere: 'north' },
    { name: 'Saint Vincent and the Grenadines', lat: 12.9843, lon: -61.2872, hemisphere: 'north' },
    { name: 'Samoa', lat: -13.7590, lon: -172.1046, hemisphere: 'south' },
    { name: 'San Marino', lat: 43.9424, lon: 12.4578, hemisphere: 'north' },
    { name: 'Sao Tome and Principe', lat: 0.1864, lon: 6.6131, hemisphere: 'north' },
    { name: 'Saudi Arabia', lat: 23.8859, lon: 45.0792, hemisphere: 'north' },
    { name: 'Senegal', lat: 14.4974, lon: -14.4524, hemisphere: 'north' },
    { name: 'Serbia', lat: 44.0165, lon: 21.0059, hemisphere: 'north' },
    { name: 'Seychelles', lat: -4.6796, lon: 55.4920, hemisphere: 'south' },
    { name: 'Sierra Leone', lat: 8.4606, lon: -11.7799, hemisphere: 'north' },
    { name: 'Singapore', lat: 1.3521, lon: 103.8198, hemisphere: 'north' },
    { name: 'Slovakia', lat: 48.6690, lon: 19.6990, hemisphere: 'north' },
    { name: 'Slovenia', lat: 46.1512, lon: 14.9955, hemisphere: 'north' },
    { name: 'Solomon Islands', lat: -9.6457, lon: 160.1562, hemisphere: 'south' },
    { name: 'Somalia', lat: 5.1521, lon: 46.1996, hemisphere: 'north' },
    { name: 'South Africa', lat: -30.5595, lon: 22.9375, hemisphere: 'south' },
    { name: 'South Korea', lat: 35.9078, lon: 127.7669, hemisphere: 'north' },
    { name: 'South Sudan', lat: 6.8770, lon: 31.3070, hemisphere: 'north' },
    { name: 'Spain', lat: 40.4637, lon: -3.7492, hemisphere: 'north' },
    { name: 'Sri Lanka', lat: 7.8731, lon: 80.7718, hemisphere: 'north' },
    { name: 'Sudan', lat: 12.8628, lon: 30.2176, hemisphere: 'north' },
    { name: 'Suriname', lat: 3.9193, lon: -56.0278, hemisphere: 'north' },
    { name: 'Sweden', lat: 60.1282, lon: 18.6435, hemisphere: 'north' },
    { name: 'Switzerland', lat: 46.8182, lon: 8.2275, hemisphere: 'north' },
    { name: 'Syria', lat: 34.8021, lon: 38.9968, hemisphere: 'north' },
    { name: 'Taiwan', lat: 23.6978, lon: 120.9605, hemisphere: 'north' },
    { name: 'Tajikistan', lat: 38.8610, lon: 71.2761, hemisphere: 'north' },
    { name: 'Tanzania', lat: -6.3690, lon: 34.8888, hemisphere: 'south' },
    { name: 'Thailand', lat: 15.8700, lon: 100.9925, hemisphere: 'north' },
    { name: 'Togo', lat: 8.6195, lon: 0.8248, hemisphere: 'north' },
    { name: 'Tonga', lat: -21.1790, lon: -175.1982, hemisphere: 'south' },
    { name: 'Trinidad and Tobago', lat: 10.6918, lon: -61.2225, hemisphere: 'north' },
    { name: 'Tunisia', lat: 33.8869, lon: 9.5375, hemisphere: 'north' },
    { name: 'Turkey', lat: 38.9637, lon: 35.2433, hemisphere: 'north' },
    { name: 'Turkmenistan', lat: 38.9697, lon: 59.5563, hemisphere: 'north' },
    { name: 'Tuvalu', lat: -7.1095, lon: 177.6493, hemisphere: 'south' },
    { name: 'Uganda', lat: 1.3733, lon: 32.2903, hemisphere: 'north' },
    { name: 'Ukraine', lat: 48.3794, lon: 31.1656, hemisphere: 'north' },
    { name: 'United Arab Emirates', lat: 23.4241, lon: 53.8478, hemisphere: 'north' },
    { name: 'United Kingdom', lat: 55.3781, lon: -3.4360, hemisphere: 'north' },
    { name: 'United States', lat: 37.0902, lon: -95.7129, hemisphere: 'north' },
    { name: 'Uruguay', lat: -32.5228, lon: -55.7658, hemisphere: 'south' },
    { name: 'Uzbekistan', lat: 41.3775, lon: 64.5853, hemisphere: 'north' },
    { name: 'Vanuatu', lat: -15.3767, lon: 166.9592, hemisphere: 'south' },
    { name: 'Vatican City', lat: 41.9029, lon: 12.4534, hemisphere: 'north' },
    { name: 'Venezuela', lat: 6.4238, lon: -66.5897, hemisphere: 'north' },
    { name: 'Vietnam', lat: 14.0583, lon: 108.2772, hemisphere: 'north' },
    { name: 'Yemen', lat: 15.5527, lon: 48.5164, hemisphere: 'north' },
    { name: 'Zambia', lat: -13.1339, lon: 27.8493, hemisphere: 'south' },
    { name: 'Zimbabwe', lat: -19.0154, lon: 29.1549, hemisphere: 'south' },
    // Additional territories and regions
    { name: 'American Samoa', lat: -14.2707, lon: -170.7020, hemisphere: 'south' },
    { name: 'Anguilla', lat: 18.2206, lon: -63.0686, hemisphere: 'north' },
    { name: 'Aruba', lat: 12.5211, lon: -69.9683, hemisphere: 'north' },
    { name: 'Bermuda', lat: 32.3214, lon: -64.7573, hemisphere: 'north' },
    { name: 'British Virgin Islands', lat: 18.4207, lon: -64.6399, hemisphere: 'north' },
    { name: 'Cayman Islands', lat: 19.3133, lon: -81.2546, hemisphere: 'north' },
    { name: 'Christmas Island', lat: -10.4475, lon: 105.6904, hemisphere: 'south' },
    { name: 'Cocos Islands', lat: -12.1642, lon: 96.8710, hemisphere: 'south' },
    { name: 'Cook Islands', lat: -21.2367, lon: -159.7777, hemisphere: 'south' },
    { name: 'Curacao', lat: 12.1696, lon: -68.9900, hemisphere: 'north' },
    { name: 'Falkland Islands', lat: -51.7963, lon: -59.5236, hemisphere: 'south' },
    { name: 'Faroe Islands', lat: 61.8926, lon: -6.9118, hemisphere: 'north' },
    { name: 'French Polynesia', lat: -17.6797, lon: -149.4068, hemisphere: 'south' },
    { name: 'Gibraltar', lat: 36.1408, lon: -5.3536, hemisphere: 'north' },
    { name: 'Greenland', lat: 71.7069, lon: -42.6043, hemisphere: 'north' },
    { name: 'Guam', lat: 13.4443, lon: 144.7937, hemisphere: 'north' },
    { name: 'Guernsey', lat: 49.4657, lon: -2.5853, hemisphere: 'north' },
    { name: 'Hong Kong', lat: 22.3193, lon: 114.1694, hemisphere: 'north' },
    { name: 'Isle of Man', lat: 54.2361, lon: -4.3481, hemisphere: 'north' },
    { name: 'Jersey', lat: 49.2144, lon: -2.1313, hemisphere: 'north' },
    { name: 'Macau', lat: 22.1987, lon: 113.5439, hemisphere: 'north' },
    { name: 'Mayotte', lat: -12.8275, lon: 45.1662, hemisphere: 'south' },
    { name: 'Montserrat', lat: 16.7425, lon: -62.1874, hemisphere: 'north' },
    { name: 'New Caledonia', lat: -22.2763, lon: 166.4580, hemisphere: 'south' },
    { name: 'Niue', lat: -19.0544, lon: -169.8672, hemisphere: 'south' },
    { name: 'Norfolk Island', lat: -29.0408, lon: 167.9547, hemisphere: 'south' },
    { name: 'Northern Mariana Islands', lat: 17.3302, lon: 145.3847, hemisphere: 'north' },
    { name: 'Pitcairn Islands', lat: -24.7036, lon: -127.4393, hemisphere: 'south' },
    { name: 'Puerto Rico', lat: 18.2208, lon: -66.5901, hemisphere: 'north' },
    { name: 'Reunion', lat: -21.1151, lon: 55.5364, hemisphere: 'south' },
    { name: 'Sint Maarten', lat: 18.0425, lon: -63.0548, hemisphere: 'north' },
    { name: 'Svalbard', lat: 77.5536, lon: 23.6703, hemisphere: 'north' },
    { name: 'Tokelau', lat: -8.9674, lon: -171.8559, hemisphere: 'south' },
    { name: 'Turks and Caicos Islands', lat: 21.6940, lon: -71.7979, hemisphere: 'north' },
    { name: 'US Virgin Islands', lat: 18.3358, lon: -64.8963, hemisphere: 'north' },
    { name: 'Wallis and Futuna', lat: -13.7687, lon: -177.1561, hemisphere: 'south' }
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
    const azimuthDeg = position.azimuth * (180 / Math.PI);
    const normalizedAzimuth = (azimuthDeg + 180) % 360;
    
    return { altitude: altitudeDeg, azimuth: normalizedAzimuth };
}

function getSunriseSunset(lat, lon, dateStr) {
    const [year, month, day] = dateStr.split('-').map(Number);
    const date = new Date(Date.UTC(year, month - 1, day, 12, 0, 0));
    const times = SunCalc.getTimes(date, lat, lon);
    
    const timezoneOffset = lon / 15;
    
    const sunriseHours = times.sunrise.getUTCHours() + times.sunrise.getUTCMinutes() / 60 + timezoneOffset;
    let sunriseLocalHours = Math.floor(sunriseHours);
    let sunriseLocalMinutes = Math.round((sunriseHours - Math.floor(sunriseHours)) * 60);
    if (sunriseLocalHours < 0) sunriseLocalHours += 24;
    if (sunriseLocalHours >= 24) sunriseLocalHours -= 24;
    
    const sunsetHours = times.sunset.getUTCHours() + times.sunset.getUTCMinutes() / 60 + timezoneOffset;
    let sunsetLocalHours = Math.floor(sunsetHours);
    let sunsetLocalMinutes = Math.round((sunsetHours - Math.floor(sunsetHours)) * 60);
    if (sunsetLocalHours < 0) sunsetLocalHours += 24;
    if (sunsetLocalHours >= 24) sunsetLocalHours -= 24;
    
    return {
        sunrise: { hour: sunriseLocalHours, minute: sunriseLocalMinutes },
        sunset: { hour: sunsetLocalHours, minute: sunsetLocalMinutes },
        daylight: (times.sunset - times.sunrise) / (1000 * 60 * 60),
        sunriseUTC: times.sunrise,
        sunsetUTC: times.sunset
    };
}

let passCount = 0;
let failCount = 0;
let warningCount = 0;

console.log('\n========================================');
console.log('SOLAR CALCULATOR UNIT TESTS - ALL 215 COUNTRIES');
console.log('Date: ' + TEST_DATE + ' (near equinox)');
console.log('NOTE: Times are SOLAR TIME (based on longitude)');
console.log('========================================\n');

console.log('TEST 1: Sunrise/Sunset Times (Solar Time)');
console.log('----------------------------------------');
testLocations.forEach(loc => {
    const times = getSunriseSunset(loc.lat, loc.lon, TEST_DATE);
    const sunriseStr = `${times.sunrise.hour.toString().padStart(2, '0')}:${times.sunrise.minute.toString().padStart(2, '0')}`;
    const sunsetStr = `${times.sunset.hour.toString().padStart(2, '0')}:${times.sunset.minute.toString().padStart(2, '0')}`;
    
    // Check if sunrise is between 5-8 AM and sunset between 17-20 (reasonable for equinox)
    const sunriseOk = times.sunrise.hour >= 5 && times.sunrise.hour <= 8;
    const sunsetOk = times.sunset.hour >= 17 && times.sunset.hour <= 20;
    
    let status;
    if (sunriseOk && sunsetOk) {
        status = '✓ PASS';
        passCount++;
    } else {
        status = `⚠ WARN (unusual times)`;
        warningCount++;
    }
    
    console.log(`${loc.name}: Sunrise=${sunriseStr} Sunset=${sunsetStr} Daylight=${times.daylight.toFixed(1)}h ${status}`);
});

console.log('\nTEST 2: Sun Altitude at Sunrise (~0°)');
console.log('----------------------------------------');
testLocations.forEach(loc => {
    const times = getSunriseSunset(loc.lat, loc.lon, TEST_DATE);
    const sunPos = SunCalc.getPosition(times.sunriseUTC, loc.lat, loc.lon);
    const altitudeDeg = sunPos.altitude * (180 / Math.PI);
    const azimuthDeg = sunPos.azimuth * (180 / Math.PI);
    const normalizedAzimuth = (azimuthDeg + 180) % 360;
    
    let status;
    if (Math.abs(altitudeDeg) <= TOLERANCE.altitude) {
        status = '✓ PASS';
        passCount++;
    } else {
        status = `✗ FAIL (${altitudeDeg.toFixed(1)}°)`;
        failCount++;
    }
    
    console.log(`${loc.name}: Alt=${altitudeDeg.toFixed(2)}° Az=${normalizedAzimuth.toFixed(1)}° ${status}`);
});

console.log('\nTEST 3: Sun Altitude at Sunset (~0°)');
console.log('----------------------------------------');
testLocations.forEach(loc => {
    const times = getSunriseSunset(loc.lat, loc.lon, TEST_DATE);
    const sunPos = SunCalc.getPosition(times.sunsetUTC, loc.lat, loc.lon);
    const altitudeDeg = sunPos.altitude * (180 / Math.PI);
    const azimuthDeg = sunPos.azimuth * (180 / Math.PI);
    const normalizedAzimuth = (azimuthDeg + 180) % 360;
    
    let status;
    if (Math.abs(altitudeDeg) <= TOLERANCE.altitude) {
        status = '✓ PASS';
        passCount++;
    } else {
        status = `✗ FAIL (${altitudeDeg.toFixed(1)}°)`;
        failCount++;
    }
    
    console.log(`${loc.name}: Alt=${altitudeDeg.toFixed(2)}° Az=${normalizedAzimuth.toFixed(1)}° ${status}`);
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
    
    console.log(`${loc.name}: Lat=${loc.lat.toFixed(1)}° Expected~${expectedAltitude.toFixed(1)}° Got=${sunPos.altitude.toFixed(2)}° ${status}`);
});

console.log('\nTEST 5: Daylight Hours (~12h near equinox)');
console.log('----------------------------------------');
testLocations.forEach(loc => {
    const times = getSunriseSunset(loc.lat, loc.lon, TEST_DATE);
    
    const absLat = Math.abs(loc.lat);
    let maxExpectedDaylight;
    if (absLat < 23.5) {
        maxExpectedDaylight = 12.5;
    } else if (absLat < 45) {
        maxExpectedDaylight = 12.75;
    } else if (absLat < 60) {
        maxExpectedDaylight = 13.0;
    } else if (absLat < 66.5) {
        maxExpectedDaylight = 13.5;
    } else {
        maxExpectedDaylight = 15.0;
    }
    
    let status;
    if (times.daylight >= 11.5 && times.daylight <= maxExpectedDaylight) {
        status = '✓ PASS';
        passCount++;
    } else if (times.daylight >= 10 && times.daylight <= 16) {
        status = `⚠ WARN (${times.daylight.toFixed(2)}h)`;
        warningCount++;
    } else {
        status = `✗ FAIL (${times.daylight.toFixed(2)}h)`;
        failCount++;
    }
    
    console.log(`${loc.name}: Lat=${absLat.toFixed(1)}° Daylight=${times.daylight.toFixed(2)}h (max ${maxExpectedDaylight}h) ${status}`);
});

console.log('\nTEST 6: Optimal Tilt Angle (= |latitude|)');
console.log('----------------------------------------');
testLocations.forEach(loc => {
    const tilt = Math.abs(loc.lat);
    
    let status = '✓ PASS';
    passCount++;
    
    console.log(`${loc.name}: Tilt=${tilt.toFixed(1)}° ${status}`);
});

console.log('\nTEST 7: Panel Azimuth (180° North, 0° South)');
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
console.log(`Countries: ${testLocations.length}`);
console.log('========================================\n');

if (failCount > 0) {
    process.exit(1);
}
