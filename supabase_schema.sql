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

-- Create Analytics Table (V4: Deep Observability)
create table if not exists analytics (
  id text primary key,
  hashed_key text references api_keys(hashed_key),
  session_id text,
  model text,
  tokens_in int,
  tokens_out int,
  tokens_saved int,
  latency_ms double precision,
  cost_saved_usd double precision,
  total_cost_usd double precision,
  reconstruction_log jsonb,
  timestamp double precision,
  or_id text,
  
  -- Observability Extensions
  raw_messages jsonb, -- The exact messages received by the gateway
  response_message jsonb, -- The exact response from OpenRouter
  optimized_prompt text -- The final prompt that was actually sent (if optimized)
);

-- Create Filtered Index for Faster Stats Aggregation
create index if not exists idx_analytics_key_timestamp on analytics(hashed_key, timestamp desc);
create index if not exists idx_analytics_session on analytics(session_id);

-- Create Sessions Table (Replaces Redis)
create table if not exists sessions (
  session_id text primary key,
  data jsonb not null, -- Stores the compressed history + atom list
  created_at double precision,
  updated_at double precision
);
