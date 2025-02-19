import uvicorn

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

from src.embedding import RAG
from src.settings import Settings

from api.models import QueryRequest

app = FastAPI()
router = APIRouter()

setting = Settings()
rag = RAG(setting)


@router.get("/")
async def get_root() -> str:
    return JSONResponse(content={"message": "Hello World"})


@router.post("/query")
async def get_query(query_request: QueryRequest) -> str:
    res, filenames, results_content = rag.contextual_rag_search(
        query_request.content, debug=True
    )
    print(f"result: {res}")
    return JSONResponse(content={"result": res})


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
