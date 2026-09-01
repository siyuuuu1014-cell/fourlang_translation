from __future__ import annotations
import argparse
import uvicorn

def parse_args():
    p = argparse.ArgumentParser(description="Run the FourLang FastAPI inference service.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    uvicorn.run(
        "service.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,
    )

if __name__ == "__main__":
    main()
