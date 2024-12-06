for pl in 96 192 336 720; do
  python -m cli.eval \
    run_name=lsf_eval \
    model=moirai_1.0_R_large \
    model.patch_size=32 \
    model.context_length=5000 \
    data=lsf_test \
    data.dataset_name=ETTh1 \
    data.mode=S \
    data.prediction_length=$pl
done

for pl in 96 192 336 720; do
  python -m cli.eval \
    run_name=lsf_eval \
    model=moirai_1.0_R_large \
    model.patch_size=64 \
    model.context_length=2000 \
    data=lsf_test \
    data.dataset_name=ETTh2 \
    data.mode=S \
    data.prediction_length=$pl
done

for pl in 96 192 336 720; do
  python -m cli.eval \
    run_name=lsf_eval \
    model=moirai_1.0_R_large \
    model.patch_size=128 \
    model.context_length=5000 \
    data=lsf_test \
    data.dataset_name=ETTm1 \
    data.mode=S \
    data.prediction_length=$pl
done

for pl in 96 192 336 720; do
  python -m cli.eval \
    run_name=lsf_eval \
    model=moirai_1.0_R_large \
    model.patch_size=64 \
    model.context_length=4000 \
    data=lsf_test \
    data.dataset_name=ETTm2 \
    data.mode=M \
    data.prediction_length=$pl
done

for pl in 96 192 336 720; do
  python -m cli.eval \
    run_name=lsf_eval \
    model=moirai_1.0_R_large \
    model.patch_size=32 \
    model.context_length=5000 \
    data=lsf_test \
    data.dataset_name=electricity \
    data.mode=S \
    data.prediction_length=$pl
done

for pl in 96 192 336 720; do
  python -m cli.eval \
    run_name=lsf_eval \
    model=moirai_1.0_R_large \
    model.patch_size=128 \
    model.context_length=5000 \
    data=lsf_test \
    data.dataset_name=weather \
    data.mode=S \
    data.prediction_length=$pl
done