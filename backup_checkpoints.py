"""
학습이 끝난 checkpoints/ 폴더를 Hugging Face Hub에 통째로 백업하는 스크립트.
레포가 없으면 자동으로 만들고(exist_ok), 있으면 그냥 업로드합니다.

사용법 (vast.ai 인스턴스, 레포 루트에서):
    hf auth login                       # 최초 1회, Write 권한 토큰으로
    python backup_checkpoints.py --repo_id snnipe/baseline3-checkpoints
    python backup_checkpoints.py --repo_id CUAI-CV-Team4/ours-checkpoints --local_dir checkpoints
"""
import os
import argparse
from huggingface_hub import HfApi, create_repo


def backup(local_dir, repo_id, repo_type="model", private=True):
    if not os.path.isdir(local_dir):
        raise SystemExit(f"❌ '{local_dir}' 폴더가 없습니다. 학습이 끝났는지, --local_dir 경로가 맞는지 확인하세요.")

    files = os.listdir(local_dir)
    if not files:
        raise SystemExit(f"❌ '{local_dir}' 폴더가 비어있습니다. 저장된 체크포인트가 없습니다.")

    print(f"📦 백업 대상: {local_dir}/ ({len(files)}개 파일)")
    print(f"🔧 레포 확인/생성: {repo_id} (private={private})")
    create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)

    print("⬆️  업로드 시작...")
    api = HfApi()
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"✅ 백업 완료: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="checkpoints 폴더를 Hugging Face Hub에 백업")
    parser.add_argument("--local_dir", default="checkpoints", help="백업할 로컬 폴더 (기본: checkpoints)")
    parser.add_argument("--repo_id", required=True, help="예: snnipe/baseline3-checkpoints 또는 CUAI-CV-Team4/ours-checkpoints")
    parser.add_argument("--public", action="store_true", help="이 옵션을 주면 public 레포로 생성 (기본은 private)")
    args = parser.parse_args()

    backup(args.local_dir, args.repo_id, private=not args.public)
