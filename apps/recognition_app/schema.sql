create table if not exists public.diagnosis_history (
  id uuid primary key,
  predicted_label text not null,
  disease_name text not null,
  confidence double precision not null,
  model_name text not null,
  image_path text,
  created_at timestamptz not null default now()
);

alter table public.diagnosis_history enable row level security;
