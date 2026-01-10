-- Migration to ensure Storage RLS policies for 'snapshots' bucket are correct
-- Run this in the Supabase SQL Editor

-- 1. Create the bucket if it doesn't exist, and ensure it's public
insert into storage.buckets (id, name, public)
values ('snapshots', 'snapshots', true)
on conflict (id) do update set public = true;

-- 2. Enable RLS on objects (Skipped: usually enabled by default, and user might not have permission to alter system tables)
-- alter table storage.objects enable row level security;

-- 3. Drop existing policies to avoid conflicts
drop policy if exists "Authenticated users can upload snapshots" on storage.objects;
drop policy if exists "Public can view snapshots" on storage.objects;
drop policy if exists "Users can delete own snapshots" on storage.objects;
drop policy if exists "Give users access to own folder 1ok12c_0" on storage.objects;
drop policy if exists "Give users access to own folder 1ok12c_1" on storage.objects;
drop policy if exists "Give users access to own folder 1ok12c_2" on storage.objects;
drop policy if exists "Give users access to own folder 1ok12c_3" on storage.objects;


-- 4. Re-create Policy: Authenticated users can upload to their own folder
-- This allows INSERT if the user is authenticated and the file path starts with their user ID.
create policy "Authenticated users can upload snapshots"
on storage.objects for insert
to authenticated
with check (
  bucket_id = 'snapshots' and
  (storage.foldername(name))[1] = auth.uid()::text
);

-- 5. Re-create Policy: Everyone can view snapshots (public bucket)
-- This allows SELECT for everyone (public) for files in the snapshots bucket.
create policy "Public can view snapshots"
on storage.objects for select
to public
using (bucket_id = 'snapshots');

-- 6. Re-create Policy: Users can delete their own snapshots
create policy "Users can delete own snapshots"
on storage.objects for delete
to authenticated
using (
  bucket_id = 'snapshots' and
  (storage.foldername(name))[1] = auth.uid()::text
);

-- 7. NOTIFY
-- If you are seeing errors, ensure that your application code is uploading to:
-- snapshots/{user_id}/{filename}
