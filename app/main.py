from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv
from schema import AddContentRequest, DeleteContentRequest, InvokeRequest
from kb import KnowledgeBaseManager
load_dotenv()

# Initialize FastAPI app
app = FastAPI(redirect_slashes=False)

kb_manager = KnowledgeBaseManager()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Health check route
@app.get("/")
def read_root():
    return PlainTextResponse(content="Healthy", status_code=200)



# Add content to knowledge base
@app.post("/api/add")
async def add_knowledge(request: AddContentRequest):
    return JSONResponse(
        content={"message": "Content added", "content":request.content},
        status_code=200
    )


# Delete content from knowledge base
@app.post("/api/delete")
async def delete_knowledge(request: DeleteContentRequest):

    return JSONResponse(
        content={"message": "Content deleted", "content": request.ids},
        status_code=200
    )


# Invoke: find similar content from query
@app.post("/api/invoke")
async def invoke(request: InvokeRequest):

    return JSONResponse(
        content={"message": "Content search", "content": request.content},
        status_code=200
    )