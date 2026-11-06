---
name: schema-first-sql
description: Verify table and column names before running SQL.
version: 1.0.0
author: Stan + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [sql, postgresql, schema, verification]
    category: devops
---

# Schema-First SQL Skill

Run one schema lookup before executing any SQL against a database you did not
just inspect in this session. Never write table or column names from memory;
schema drift and near-miss names (`equity` vs `total_equity`) are the most
common recurring production defect class.

## When to Use

- Before the first `SELECT`, `INSERT`, `UPDATE`, or `DELETE` against a table
  in this session — including inside cron jobs and delegated tasks.
- After any schema-changing migration, deploy, or restore.
- Immediately when PostgreSQL answers `column "x" does not exist` or
  `relation "y" does not exist`: stop, look up the real name, then fix the
  statement. Do not retry a second guess.

## Prerequisites

- Database access via the `terminal` tool (`psql` or an equivalent client).
- Connection settings come from the deployment's own environment or config;
  never hardcode hosts, ports, or credentials in commands you compose.

## How to Run

One lookup per table, then execute:

```
# 1. Confirm the table exists and see its real columns:
\d "paper_positions"

# 2. Or query the catalog directly (works in scripts too):
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'paper_positions'
ORDER BY ordinal_position;

# 3. Only now run the intended statement.
```

On the OntoX monorepo deployment, prefer the authoritative snapshot tool which
covers every database in one read-only pass:

```
python3 tools/devops/schema_snapshot.py --db loom --table af_event
```

## Quick Reference

| Situation | Correct move |
|---|---|
| Unsure a table exists | `\dt pattern` or query `pg_tables` first |
| Unsure of a column name | `\d table` or `information_schema.columns` first |
| Table "not found" but you saw it before | Check which database you are connected to (`\c` list, `\l`); cross-database typos recur |
| Writing SQL for a cron job | Paste the verified column list into the job prompt |

## Procedure

1. Name the database and table the statement will touch.
2. Run one schema lookup (`\d`, catalog query, or snapshot tool) through the
   `terminal` tool and read the actual column list.
3. Rewrite the SQL using only names that appeared in that output.
4. Execute once. If PostgreSQL reports a missing column or relation, return
   to step 2 — a second guessed name is not a fix.
5. For recurring jobs, pin the verified schema (or the snapshot command) next
   to the job definition so the next session inherits the facts.

## Failure Patterns (Real Incidents)

- `SELECT equity FROM paper_equity` failed 15 nights in a row on a cron job;
  the real column was `total_equity`. One `\d` lookup would have prevented
  every occurrence.
- `af_event` probes failed because the table lives in the `loom` database,
  not the one the probe connected to. Schema lookup and database lookup are
  two different checks — do both.
