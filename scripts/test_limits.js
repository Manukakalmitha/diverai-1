import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
dotenv.config();

// Usage: node scripts/test_limits.js <email> <password>

const supabaseUrl = process.env.VITE_SUPABASE_URL;
const supabaseKey = process.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseKey) {
    console.error("Please ensure .env contains VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY");
    process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

async function runTest() {
    const email = process.argv[2];
    const password = process.argv[3];

    if (!email || !password) {
        console.error("Usage: node scripts/test_limits.js <email> <password>");
        process.exit(1);
    }

    console.log(`Authenticating as ${email}...`);
    const { data: { session }, error: authError } = await supabase.auth.signInWithPassword({
        email,
        password
    });

    if (authError || !session) {
        console.error("Login failed:", authError?.message);
        process.exit(1);
    }

    console.log("Logged in. Starting Rate Limit Test (15 requests)...");

    let success = 0;
    let limited = 0;
    let errors = 0;

    for (let i = 0; i < 15; i++) {
        const start = Date.now();
        // Use a dummy base64 image
        const { data, error } = await supabase.functions.invoke('detect_ticker', {
            body: { image: 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAAAAAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEGMgBAAAAAAAAB/8QAFgABAQEAAAAAAAAAAAAAAAAAAAcJ/8QAFhABAQEAAAAAAAAAAAAAAAAAAAAB/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AP/Z' },
            headers: { Authorization: `Bearer ${session.access_token}` }
        });

        const duration = Date.now() - start;

        if (error) {
            // supabase-js returns error object for 4xx/5xx if access is via functions.invoke? 
            // Actually invoke returns { data, error }. error is populated on failure.
            if (error && error.context && error.context.status === 429) {
                console.log(`req ${i + 1}: 429 TOO MANY REQUESTS (${duration}ms) - EXPECTED`);
                limited++;
            } else {
                console.log(`req ${i + 1}: ERROR ${error.message} (${duration}ms)`);
                errors++;
            }
        } else {
            console.log(`req ${i + 1}: SUCCESS (${duration}ms)`);
            success++;
        }
    }

    console.log("\n--- REPORT ---");
    console.log(`Successful: ${success}`);
    console.log(`Rate Limited: ${limited}`);
    console.log(`Other Errors: ${errors}`);

    if (limited > 0 && success > 0) {
        console.log("✅ Verification PASSED: Rate limits are active.");
    } else if (limited === 0) {
        console.log("❌ Verification FAILED: No rate limiting observed (or limit too high).");
    } else {
        console.log("⚠️ Verification INCONCLUSIVE.");
    }
}

runTest();
