from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from orchestrator.persistence.common import PdfChunkRow, json_dumps, json_loads, utc_now_iso

if TYPE_CHECKING:
    from orchestrator.persistence.db import Database


class PdfRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def upsert_pdf_document(
        self,
        path: str,
        content_hash: str,
        mtime: float,
        page_count: int,
    ) -> tuple[int, bool]:
        async with self.db._lock:
            row = await self.db._fetchone(
                "SELECT id, content_hash FROM pdf_documents WHERE path = ?",
                (path,),
            )
            now = utc_now_iso()
            if row is None:
                cursor = await self.db.conn.execute(
                    """
                    INSERT INTO pdf_documents(path, content_hash, mtime, page_count, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (path, content_hash, mtime, page_count, now),
                )
                await self.db.conn.commit()
                return int(cursor.lastrowid), True
            changed = row["content_hash"] != content_hash
            await self.db.conn.execute(
                """
                UPDATE pdf_documents
                SET content_hash = ?, mtime = ?, page_count = ?, updated_at = ?
                WHERE path = ?
                """,
                (content_hash, mtime, page_count, now, path),
            )
            await self.db.conn.commit()
            return int(row["id"]), changed

    async def remove_pdf_document(self, path: str) -> None:
        async with self.db._lock:
            row = await self.db._fetchone("SELECT id FROM pdf_documents WHERE path = ?", (path,))
            if not row:
                return
            doc_id = int(row["id"])
            await self.db.conn.execute("DELETE FROM pdf_chunks_fts WHERE path = ?", (path,))
            await self.db.conn.execute("DELETE FROM pdf_chunks WHERE document_id = ?", (doc_id,))
            await self.db.conn.execute("DELETE FROM pdf_documents WHERE id = ?", (doc_id,))
            await self.db.conn.commit()

    async def replace_pdf_chunks(
        self,
        document_id: int,
        path: str,
        chunks: Iterable[tuple[int, int, str]],
    ) -> list[int]:
        async with self.db._lock:
            existing = await self.db._fetchall(
                "SELECT id FROM pdf_chunks WHERE document_id = ?",
                (document_id,),
            )
            existing_ids = [int(r["id"]) for r in existing]
            if existing_ids:
                await self.db.conn.executemany(
                    "DELETE FROM pdf_embeddings WHERE chunk_id = ?",
                    [(cid,) for cid in existing_ids],
                )
            await self.db.conn.execute("DELETE FROM pdf_chunks_fts WHERE path = ?", (path,))
            await self.db.conn.execute("DELETE FROM pdf_chunks WHERE document_id = ?", (document_id,))
            created_ids: list[int] = []
            for chunk_index, page_number, text in chunks:
                cursor = await self.db.conn.execute(
                    """
                    INSERT INTO pdf_chunks(document_id, chunk_index, page_number, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (document_id, chunk_index, page_number, text),
                )
                chunk_id = int(cursor.lastrowid)
                created_ids.append(chunk_id)
                await self.db.conn.execute(
                    "INSERT INTO pdf_chunks_fts(chunk_id, path, text) VALUES (?, ?, ?)",
                    (chunk_id, path, text),
                )
            await self.db.conn.commit()
            return created_ids

    async def set_chunk_embeddings(self, vectors: Iterable[tuple[int, list[float]]]) -> None:
        async with self.db._lock:
            for chunk_id, vector in vectors:
                await self.db.conn.execute(
                    """
                    INSERT INTO pdf_embeddings(chunk_id, vector_json)
                    VALUES (?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET vector_json = excluded.vector_json
                    """,
                    (chunk_id, json_dumps(vector)),
                )
            await self.db.conn.commit()

    async def list_known_pdf_paths(self) -> list[str]:
        rows = await self.db._fetchall("SELECT path FROM pdf_documents")
        return [str(r["path"]) for r in rows]

    async def fts_search(self, query: str, top_k: int, pdf_scope: list[str] | None = None) -> list[PdfChunkRow]:
        if not query.strip():
            return []
        params: list[Any] = [query]
        sql = """
            SELECT c.id AS chunk_id, d.path AS document_path, c.page_number, c.chunk_index, c.text, e.vector_json
            FROM pdf_chunks_fts f
            JOIN pdf_chunks c ON c.id = CAST(f.chunk_id AS INTEGER)
            JOIN pdf_documents d ON d.id = c.document_id
            LEFT JOIN pdf_embeddings e ON e.chunk_id = c.id
            WHERE pdf_chunks_fts MATCH ?
        """
        if pdf_scope:
            placeholders = ",".join("?" for _ in pdf_scope)
            sql += f" AND d.path IN ({placeholders})"
            params.extend(pdf_scope)
        sql += " ORDER BY bm25(pdf_chunks_fts) LIMIT ?"
        params.append(top_k)
        rows = await self.db._fetchall(sql, params)
        return [
            PdfChunkRow(
                chunk_id=int(row["chunk_id"]),
                document_path=str(row["document_path"]),
                page_number=int(row["page_number"]),
                chunk_index=int(row["chunk_index"]),
                text=str(row["text"]),
                embedding=json_loads(row["vector_json"], None),
            )
            for row in rows
        ]

    async def vector_candidates(self, pdf_scope: list[str] | None = None) -> list[PdfChunkRow]:
        params: list[Any] = []
        sql = """
            SELECT c.id AS chunk_id, d.path AS document_path, c.page_number, c.chunk_index, c.text, e.vector_json
            FROM pdf_chunks c
            JOIN pdf_documents d ON d.id = c.document_id
            JOIN pdf_embeddings e ON e.chunk_id = c.id
        """
        if pdf_scope:
            placeholders = ",".join("?" for _ in pdf_scope)
            sql += f" WHERE d.path IN ({placeholders})"
            params.extend(pdf_scope)
        rows = await self.db._fetchall(sql, params)
        return [
            PdfChunkRow(
                chunk_id=int(row["chunk_id"]),
                document_path=str(row["document_path"]),
                page_number=int(row["page_number"]),
                chunk_index=int(row["chunk_index"]),
                text=str(row["text"]),
                embedding=json_loads(row["vector_json"], None),
            )
            for row in rows
        ]

    async def get_pdf_path_hashes(self) -> dict[str, str]:
        rows = await self.db._fetchall("SELECT path, content_hash FROM pdf_documents")
        return {str(r["path"]): str(r["content_hash"]) for r in rows}
