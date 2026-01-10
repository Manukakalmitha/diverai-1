-- Migration to fix Storage RLS policies for Snapshots bucket

-- 1. Create the bucket if it doesn't exist, and ensure it's public
insert into storage.buckets (id, name, public)
values ('snapshots', 'snapshots', true)
on conflict (id) do update set public = true;

-- 2. Enable RLS on objects (standard practice)
alter table storage.objects enable row level security;

-- 3. Policy: Authenticated users can upload to their own folder
-- This allows INSERT if the user is authenticated and the file path starts with their user ID.
create policy "Authenticated users can upload snapshots"
on storage.objects for insert
to authenticated
with check (
  bucket_id = 'snapshots' and
  (storage.foldername(name))[1] = auth.uid()::text
);

-- 4. Policy: Everyone can view snapshots (public bucket)
-- This allows SELECT for everyone (public) for files in the snapshots bucket.
create policy "Public can view snapshots"
on storage.objects for select
to public
using (bucket_id = 'snapshots');

-- 5. Policy: Users can delete their own snapshots
create policy "Users can delete own snapshots"
on storage.objects for delete
to authenticated
using (
  bucket_id = 'snapshots' and
  (storage.foldername(name))[1] = auth.uid()::text
);
