import sys
import zipfile
from pathlib import Path

def pack_codebase(output_path: Path | str | None = None) -> Path:
    repo_dir = Path(__file__).resolve().parents[1]
    
    if output_path is None:
        if len(sys.argv) > 1:
            dest = Path(sys.argv[1])
        else:
            dest = repo_dir / "Video_RAG_Research_Codebase.zip"
    else:
        dest = Path(output_path)

    exclude_dirs = {
        ".git", ".venv", "__pycache__", ".pytest_cache", ".idea", ".vscode",
        "videos", "qdrant"  # Strictly exclude videos and qdrant folders under data/
    }

    exclude_files = {
        ".DS_Store", ".env", "Video_RAG_Research_Codebase.zip"
    }

    dest.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in repo_dir.rglob("*"):
            if path.is_dir():
                continue
            
            parts = path.relative_to(repo_dir).parts
            if any(p in exclude_dirs for p in parts):
                continue
            if path.name in exclude_files or path.name.endswith(".pyc"):
                continue
                
            arcname = path.relative_to(repo_dir)
            zf.write(path, arcname=arcname)

    print(f"Codebase zipped cleanly to: {dest} (size: {dest.stat().st_size / (1024*1024):.2f} MB)")
    return dest

if __name__ == "__main__":
    pack_codebase()
