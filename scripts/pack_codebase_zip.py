import zipfile
from pathlib import Path

repo_dir = Path("/Users/maksimlyara/Documents/GitHub/multimodal-video-search")
output_zip = Path("/Users/maksimlyara/Documents/notes/Максим/учеба/ИТМО магистратура портфолио/Junior ML Contest 2026 — 3 волна — Video-RAG Research/FINAL/Video_RAG_Research_Codebase.zip")

exclude_dirs = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".idea", ".vscode",
    "videos", "qdrant"  # Strictly exclude videos and qdrant folders under data/
}

exclude_files = {
    ".DS_Store", ".env"
}

with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in repo_dir.rglob("*"):
        if path.is_dir():
            continue
        
        # Check if path contains excluded dir
        parts = path.relative_to(repo_dir).parts
        if any(p in exclude_dirs for p in parts):
            continue
        if path.name in exclude_files or path.name.endswith(".pyc"):
            continue
            
        arcname = path.relative_to(repo_dir)
        zf.write(path, arcname=arcname)

print(f"Codebase zipped cleanly to: {output_zip} (size: {output_zip.stat().st_size / (1024*1024):.2f} MB)")
