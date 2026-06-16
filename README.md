# cofiswarm-backend-vllm

Cofiswarm component: `backend-vllm`.

- Layout: [REPO-STANDARD-LAYOUT](https://github.com/keepdevops/cofiswarmdev/blob/main/docs/REPO-STANDARD-LAYOUT.md)
- Migration: [MIGRATION-SPRINTS](https://github.com/keepdevops/cofiswarmdev/blob/main/docs/MIGRATION-SPRINTS.md)

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
