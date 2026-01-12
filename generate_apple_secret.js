const jwt = require('jsonwebtoken');
const fs = require('fs');

// 1. FILL IN YOUR DETAILS HERE
const TEAM_ID = 'YOUR_TEAM_ID'; // e.g., 8AB9328...
const KEY_ID = 'YOUR_KEY_ID';   // e.g., 9823... from the Key download
const CLIENT_ID = 'YOUR_SERVICE_ID'; // e.g., com.example.app.service
const PRIVATE_KEY_PATH = './AuthKey_XXXXXXXXXX.p8'; // Path to the downloaded .p8 file

// 2. RUN: npm install jsonwebtoken
// 3. RUN: node generate_apple_secret.js

try {
    const privateKey = fs.readFileSync(PRIVATE_KEY_PATH);

    const token = jwt.sign({}, privateKey, {
        algorithm: 'ES256',
        expiresIn: '180d', // 6 months (maximum allowed by Apple)
        audience: 'https://appleid.apple.com',
        issuer: TEAM_ID,
        subject: CLIENT_ID,
        keyid: KEY_ID,
        header: {
            alg: 'ES256',
            kid: KEY_ID
        }
    });

    console.log('\n✅ Your Apple Secret Key (valid for 6 months):');
    console.log('---------------------------------------------------');
    console.log(token);
    console.log('---------------------------------------------------\n');
    console.log('Copy this token and paste it into the "Secret Key" field in Supabase.');

} catch (error) {
    console.error('Error generating token:', error.message);
    if (error.code === 'ENOENT') {
        console.error('-> Could not find the .p8 file. Make sure PRIVATE_KEY_PATH is correct.');
    }
}
