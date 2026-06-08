"""
One-off: scrub pre-existing secrets from documents + messages rows using the
same scrub_text logic now wired into the ingest/derive paths. Idempotent.
Uses the app's own SQLAlchemy async session.
"""
import asyncio
from sqlalchemy import text
from src.dependencies import tracked_db
from src.utils.secrets import scrub_text


async def scrub_table(db, table: str) -> int:
    rows = (
        await db.execute(
            text(f"SELECT id, content FROM {table} WHERE content IS NOT NULL")
        )
    ).all()
    changed = 0
    hit_summary: dict[str, int] = {}
    for row in rows:
        res = scrub_text(row.content)
        if res.found and res.text != row.content:
            await db.execute(
                text(f"UPDATE {table} SET content = :c WHERE id = :i"),
                {"c": res.text, "i": row.id},
            )
            changed += 1
            for h in res.hit_types:
                hit_summary[h] = hit_summary.get(h, 0) + 1
    print(f"{table}: scanned {len(rows)} rows, redacted {changed}, hits={hit_summary}")
    return changed


async def main():
    async with tracked_db("scrub_existing_secrets") as db:
        total = 0
        for t in ("documents", "messages"):
            total += await scrub_table(db, t)
        await db.commit()
        print(f"TOTAL redacted: {total}")


asyncio.run(main())
