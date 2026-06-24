# cofiswarm-backend-vllm

Cofiswarm component: `backend-vllm`.

- Layout: [REPO-STANDARD-LAYOUT](https://github.com/keepdevops/cofiswarm-docs/blob/main/REPO-STANDARD-LAYOUT.md)
- Migration: [MIGRATION-SPRINTS](https://github.com/keepdevops/cofiswarm-docs/blob/main/MIGRATION-SPRINTS.md)

## FHS paths

| Path | Purpose |
|------|---------|
| `/etc/cofiswarm/backend-vllm/` | config |
| `/var/lib/cofiswarm/backend-vllm/` | state |
| `/var/log/cofiswarm/backend-vllm/` | logs |

## Test

```bash
./test/scripts/assert-layout.sh backend-vllm
```
