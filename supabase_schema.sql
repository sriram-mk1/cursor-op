-- Create API Keys Table
create table if not exists api_keys (
  hashed_key text primary key,
  raw_key text unique not null,
  name text not null,
  user_id text not null,
  created_at double precision not null,
  total_tokens_saved int default 0,
  total_requests int default 0,
  openrouter_key text
);

-- Create Conversations Table (V5: Grouping Requests)
create table if not exists conversations (
  id uuid primary key default gen_random_uuid(),
  session_id text unique not null,
  user_id text,
  hashed_key text references api_keys(hashed_key),
  title text,
  last_request_at double precision,
  metadata jsonb
);

-- Create Analytics Table (V5: Linked to Conversations)
create table if not exists analytics (
  id text primary key,
  hashed_key text references api_keys(hashed_key),
  conversation_id uuid references conversations(id),
  session_id text,
  model text,
  tokens_in int,
  tokens_out int,
  tokens_saved int,
  latency_ms double precision,
  cost_saved_usd double precision,
  total_cost_usd double precision,
  reconstruction_log jsonb, -- Stores the chunks and their scores
  timestamp double precision,
  or_id text,
  
  -- Observability Extensions
  raw_messages jsonb, -- The exact messages received
  response_message jsonb, -- The exact response
  optimized_prompt text -- The final prompt sent
);

-- Create Indexes
create index if not exists idx_analytics_convo on analytics(conversation_id);
create index if not exists idx_analytics_key_timestamp on analytics(hashed_key, timestamp desc);
create index if not exists idx_conversations_user on conversations(user_id);

-- Create Sessions Table (The "Living State")
create table if not exists sessions (
  session_id text primary key,
  data jsonb not null, 
  created_at double precision,
  updated_at double precision
);
