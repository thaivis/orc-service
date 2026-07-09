# Migration note: `middle_name` field added (passport scan)

**Status:** planned, not yet implemented.
**Affects:** `POST /scan/passport` response only. Thai ID (`/scan/thai-id`) is unaffected —
its `first_name` is always a single token, so there is nothing to split.

## What's changing

Passport MRZ carries all given names as one string (e.g. `"ADAM MICHAEL"`). Today the full
string is returned as `first_name`. Going forward it will be split on the first space:

| Field | Before | After |
|---|---|---|
| `first_name` | `"ADAM MICHAEL"` | `"ADAM"` |
| `middle_name` | *(field didn't exist)* | `"MICHAEL"` |

If there's no middle name at all, `middle_name` will be `null` and `first_name` is unchanged
(single token, same as before).

If there are 3+ given names (e.g. `"ADAM MICHAEL JOHN"`), everything after the first token goes
into `middle_name`: `first_name: "ADAM"`, `middle_name: "MICHAEL JOHN"`.

`confidence.middle_name` is added alongside the existing per-field confidence scores, using the
same score as `confidence.first_name` (they come from the same MRZ read).

## ⚠️ Breaking change

**`first_name` will no longer contain the full given-names string for passports with a middle
name.** Any code reading `first_name` and expecting `"ADAM MICHAEL"` will now get `"ADAM"`.

## What the backend team needs to do

1. Add a `middle_name: string | null` field to whatever type/DTO models the scan response
   (NestJS side).
2. Anywhere `first_name` is stored, displayed, or forwarded (e.g. saved to a guest profile,
   printed on a form) — check whether it currently assumes the full given name. If so, update
   that code path to concatenate `first_name` + `middle_name` (when present), or add a
   `middle_name` column/field to match the new shape.
3. This only affects **passport** scans. Existing Thai ID integration code needs no changes.

## Rollout

Not yet implemented — this note is to confirm the shape and give the backend team lead time
before the change ships. Ping when ready to schedule so both sides deploy in the right order.
