-- Create a table for public user profiles
create table profiles (
  id uuid references auth.users not null primary key,
  email text,
  subscription_tier text default 'free', -- 'free' or 'pro'
  upload_count int default 0,
  last_upload_date  timestamp with time zone default timezone('utc'::text, now()),
  
  constraint proper_tier check (subscription_tier in ('free', 'pro'))
);

-- Enable RLS
alter table profiles enable row level security;

-- Create policies
create policy "Public profiles are viewable by everyone." on profiles
  for select using (true);

create policy "Users can insert their own profile." on profiles
  for insert with check (auth.uid() = id);

create policy "Users can update own profile." on profiles
  for update using (auth.uid() = id);

-- Function to handle new user signup
create or replace function public.handle_new_user() 
returns trigger as $$
begin
  insert into public.profiles (id, email)
  values (new.id, new.email);
  return new;
end;
$$ language plpgsql security definer;

-- Trigger the function every time a user is created
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

-- Function to reset upload count if day has changed (Basic logic called from client or edge function)
-- For simplicity, we'll handle this logic in the client/Edge for now, or just trust the client 
-- (since this is a demo/prototype. ideal is a cron job).
