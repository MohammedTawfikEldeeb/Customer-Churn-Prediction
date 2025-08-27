from fastapi import APIRouter , Request
from routes.schemes.data import ProcessReuest
from controllers.ProcessController import ProcessController

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1", "data"]
)

@data_router.post("/process")
async def process(request: ProcessReuest):
    process_controller = ProcessController()
    return process_controller.process_data(request)

