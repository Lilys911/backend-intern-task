
from fastapi import FastAPI,HTTPException,Response
from pydantic import BaseModel, Field, field_validator

from app import llm  # noqa: F401  (you will need it)
from app.chunking import chunk_text  # noqa: F401  (you will need it)
class DocumentIn(BaseModel):
    id:str
    text:str

    @field_validator("id", "text")
    @classmethod
    def not_blank(cls,  v):
        if not v.strip():
            raise ValueError("must not be empty")
        return v
class IngestBody(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1)
class AskBody(BaseModel):
    question:str
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v):
        if not v.strip():
            raise ValueError("question must not be empty")
        return v

def create_app() -> FastAPI:
    """Return a NEW app with an EMPTY in-memory store.

    The tests call create_app() once per test, so state must not leak
    between app instances (no module-level store).
    """
    app = FastAPI()
    store: dict[str, list[dict]] = {}
    @app.get("/health")
    def health():
        return {"status": "ok"}
    @app.post("/ingest")
    def ingest(body: IngestBody):
        total_chunks = 0
        for doc in body.documents:
            pieces = chunk_text(doc.text)
            records = []
            for i, piece in enumerate(pieces):
                records.append({
                    "doc_id":doc.id,
                    "chunk_id":f"{doc.id}:{i}",
                    "text":piece,
                    "vector":llm.embed(piece),
                })
            store[doc.id] = records
            total_chunks+=len(records)
        return {"ingested": len(body.documents), "chunks":total_chunks}
    @app.post("/ask")
    def ask(body: AskBody):
        if not store:
            raise HTTPException(status_code=404, detail="no documents ingested")
        q_vec = llm.embed(body.question)

        scored = []
        for records in store.values():
            for rec in records:
                score = sum(a*b for a, b in zip(q_vec, rec["vector"]))
                scored.append((score, rec))
        scored.sort(key=lambda pair:pair[0], reverse=True)
        top = scored[:body.top_k]

        answer = llm.generate(body.question, [rec["text"] for _, rec in top])
        sources = [
            {"doc_id":rec["doc_id"], "chunk_id": rec["chunk_id"], "score":score, "text":rec["text"]}
            for score, rec in top
        ]
        return {"answer": answer, "sources":sources}
    @app.get("/documents")
    def get_documents():
        docs = [{"id":doc_id, "chunks":len(records)} for doc_id, records in store.items()]
        docs = sorted(docs, key=lambda d: d["id"])
        return  {"documents":docs}

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_documents(doc_id:str):
        if doc_id not in store:
            raise HTTPException(status_code=404, detail="document not found")
        del store[doc_id]
        return Response(status_code=204)
    # TODO: POST /ingest
    # TODO: POST /ask
    # TODO: GET /documents
    # TODO: DELETE /documents/{doc_id}

    return app


app = create_app()
