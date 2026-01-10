
import pg from 'pg';
const { Client } = pg;

const connectionString = 'postgresql://postgres.brrjoheinakfhohesogc:Kalmitha"2027@aws-1-ap-northeast-1.pooler.supabase.com:6543/postgres';

const client = new Client({
    connectionString,
    ssl: { rejectUnauthorized: false }
});

async function setupDatabase() {
    try {
        await client.connect();
        console.log('Connected to Supabase PostgreSQL...');

        const createTableQuery = `
      CREATE TABLE IF NOT EXISTS prediction_history (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        data JSONB NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW()
      );
    `;

        await client.query(createTableQuery);
        console.log('✅ Table "prediction_history" created or already exists.');

    } catch (err) {
        console.error('❌ Error setting up database:', err);
    } finally {
        await client.end();
    }
}

setupDatabase();
