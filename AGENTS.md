# AGENTS.md

Rules for anyone (human or agent) committing to this repository.

## No private or sensitive data in git, ever

- Never commit IP addresses, hostnames, VM or cluster names, SSH targets, user names, API keys,
  tokens, `.env` files, or details of the hardware and infrastructure used (GPU models, VM sizes,
  providers, internal services and ports, local LLM endpoints).
- The same applies to commit messages, branch names, PR descriptions, config comments, log excerpts,
  reports and docs. Describe compute generically ("a single GPU", "the training VM").
- Keep such details in git-ignored files only: `.env` (template in `.env.example` with placeholders),
  `logs/`, local run folders and weights (`*.pt`, `*.safetensors`). `.gitignore` is the source of truth;
  extend it before adding a new kind of local file.
- Before pushing, grep the staged diff for `10.`, `root@`, `token`, `key`, hostnames and hardware
  names; if in doubt, leave it out.
- Dataset and model cards published from `release/` follow the same rule.
