# R2 변경 내역 — Oxford-IIIT Pet / CIFAR-10 10,000장

기준: `NGV_CNN_5Fold_CODE_ONLY_R1.zip` (2026-09-29). 환경 재설치 불필요.

## 실제 변경

- `cnnlab/builder.py`: 기본 catsdogs 원본을 Oxford-IIIT Pet으로 교체. 공식 `trainval.txt` species 열로 cat=0/dog=1. 96×96 RGB 중심부 맞춤·bilinear, 1,000장/종은 유지. Oxford 원본 이미지+주석 다운로드는 약 800 MB이며 공식 MD5를 검사. 로컬 압축파일/추출 폴더도 사용 가능.
- `cnnlab/builder.py`: CIFAR-10은 1,000장/클래스, 총 10,000장. 여러 공식 학습 배치에 걸친 제한 스트림으로 필요한 후보 확보 후 중단(압축 데이터 읽기 상한 64 MiB). 후보 수/압축 선행 읽기는 최종 10,000장보다 클 수 있음. 원본 전체 자동 다운로드로 전환하지 않음.
- 기존 정상 MNIST 파일은 바이트 그대로 재사용. 기존 CIFAR 2,000장 또는 구형 고양이/개 파일은 각각 새 파일 쌍이 준비·검증된 뒤 `teacher_sources/bundle_backups/`에 백업하고 교체. 실패한 준비 단계의 기존 쌍은 유지.
- `cnnlab/data.py`: 도메인별 목표 장수, 검사/프로토콜의 수량 표기만 동적화. 80:20, 5-fold, split seed, 증강/정규화는 유지.
- `configs/exp01_cifar10.json`, `configs/exp03_models.json`: `base.train_count` 1600 → 8000만 변경. CIFAR test는 2000, fold별 6400/1600.
- `cnnlab/runner.py`: 기록 문자열의 holdout 수만 실제 시험 장수로 표기. 학습 실행 로직 변경 없음.
- `cnnlab/reporting.py`: 설명문의 고정 400/1600을 실제 수량으로 표기. 그래프 종류·수식·지표 계산은 동일.
- `00_build_data_bundle.py`, `08_make_student_zip.py`: 첫 설명문만 새 도메인별 장수/총 14000장으로 수정.
- 가이드·README·버전/상태 기록 갱신, R2 회귀시험 20개 추가. 기존 35개 시험은 수정하지 않음.

## 그대로인 파일/환경

SHA256 비교로 `install.ps1`, `requirements.txt`, `.vscode/*`, 환경 검사, 모델 구조, 학습 엔진, 공통 유틸리티, 실습 실행 파일 01~07, MNIST/고양이·개/데이터 양/하이퍼파라미터 설정이 원본과 동일함을 확인했습니다.

Python 3.12 / PyTorch 2.10.0 / torchvision 0.25.0 / NumPy 2.3.5 / scikit-learn 1.8.0 / Matplotlib 3.10.8 / Pillow 12.3.0: 버전 변경 없음.

## 기존 폴더에 패치 적용

1. 진행 중인 다운로드·학습을 종료합니다. 직접 수정한 코드는 따로 백업합니다.
2. PATCH ZIP을 임시 폴더에 풉니다. 안의 `ngv_cnn_5fold` 폴더 내용만 기존 같은 프로젝트 위치로 복사하고 동일 파일을 교체합니다. 폴더가 두 겹이 되지 않도록 합니다.
3. 기존 `.venv`, 실제 데이터, 결과는 삭제하지 않습니다. 패치에는 해당 바이너리 데이터나 설치/환경 파일이 없습니다.
4. 같은 VS Code interpreter로 `00_build_data_bundle.py` 실행 → `00_check_data.py` → 기존 01~07 실습 순서.
5. 세 데이터 검사 후 `08_make_student_zip.py`로 학생용 WITH_DATA ZIP 생성. 원본/백업/환경/결과는 학생 ZIP에서 제외됩니다.

## 검증과 한계

55개 단위·기능 시험 통과. 별도의 합성 fixture로 이전 데이터 백업/교체, MNIST 파일 보존, 무다운로드 재실행, 두 도메인 5-fold+최종재학습+시험+그래프/CSV/HTML, 학생 ZIP 구성을 검사했습니다. 합성 데이터와 그 성능은 배포하지 않습니다.

현재 CODE_ONLY ZIP에는 실제 이미지가 없습니다. 공식 웹페이지와 형식·MD5는 확인했으나 실제 원본 HTTP 다운로드는 제작 환경의 DNS 실패로 미검증입니다. Windows/GPU 실행 및 실제 데이터 정확도도 미검증입니다. `VALIDATION.json` 참조.
