-- Create table for chat messages
create table chat_messages (
  id uuid default gen_random_uuid() primary key,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  role text not null check (role in ('user', 'assistant')),
  content text not null
);

-- Enable Row Level Security (RLS)
alter table chat_messages enable row level security;

-- Create policy to allow all operations (since this is a private bot backend)
-- Alternatively, you can disable RLS if you trust the connection:
-- alter table chat_messages disable row level security;
create policy "Allow all operations for service role"
on chat_messages
for all
using (true)
with check (true);
