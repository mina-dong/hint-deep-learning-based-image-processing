# NGV CNN 5-fold / VS Code

## Data availability

Check `BUNDLE_STATUS.json` first. The original `CODE_ONLY_R2.zip` distribution **does not contain real dataset images**. External binary downloads failed in the authoring environment. A file called `WITH_DATA.zip` is generated only after MNIST 2000 / CIFAR-10 10000 / Oxford-IIIT Pet cat-dog 2000 pass validation on a machine that has the real sources.

현재 제공 CODE_ONLY 파일은 **실제 데이터 미포함**입니다. 이 점은 아직 충족하지 못한 사용자 요구입니다. 합성 이미지를 실제 MNIST/CIFAR-10/고양이·개 이미지로 대체하지 않았습니다.

## Start

1. Read `docs/Guide_KO.html` or `docs/Guide_KO.pdf`.
2. Open this folder in Windows VS Code; select Windows PowerShell for installation.
3. Run `install.ps1 -Compute cu128` through the documented PowerShell command, or choose CPU. Existing compatible Windows environments can be selected instead.
4. Select the correct Python interpreter and run `00_check_environment.py`.
5. **Teacher, CODE_ONLY distribution:** acquire sources using `00_build_data_bundle.py`. Local originals are supported. This is a required additional step because actual data is absent from the current artifact.
6. Run `00_check_data.py`. Do not start training if it fails.
7. Run `01_mnist.py`, `02_cifar10.py`, `03_catsdogs.py` for the three single-model classification labs.
8. Run `04_data_size.py`, `05_model_compare.py`, `06_hyperparameters.py` for controlled comparisons.
9. Run `07_collect_results.py` for aggregate CSV and HTML.
10. **Teacher:** run `08_make_student_zip.py` after real data is verified; distribute that resulting data-included ZIP to students.

All training scripts run directly in VS Code, without CLI arguments. Change the experiment JSON in `configs/`. All models start with random weights; training scripts make no network requests.

## Evaluation

MNIST/cat-dog: 2,000 total -> 1,600 development + 400 holdout. CIFAR-10: 10,000 total -> 8,000 development + 2,000 holdout. Five stratified folds are formed only inside the development pool. Each fold uses a fresh model/optimizer. The median validation-selected epoch is used to refit one fresh final model on the full selected development pool. Only that model is tested on the fixed domain-specific holdout. This is an internal educational split, not the official full test benchmark.

## Tests and limits

See `VALIDATION.json`. Tests use generated fixtures exclusively and do not establish accuracy on the real public datasets. The actual network download path, Windows execution, and CUDA execution were not validated in the authoring environment.

## R2 data-only changes
Oxford-IIIT Pet replaces the old cats/dogs source (the original download is about 800 MB; retained data is still 2000 images). CIFAR-10 retains exactly 10000 images, using 8000 development / 2000 holdout. Environment versions, model definitions, training logic and other configurations are unchanged. See `CHANGES_R2.md` and `docs/Guide_KO.md`. Existing data pairs are replaced only after the new pair is verified, with backups under `teacher_sources/bundle_backups/`.
