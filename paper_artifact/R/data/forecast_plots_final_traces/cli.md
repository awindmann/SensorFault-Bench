uv run python scripts/render_forecast_plots.py from-runs \
  --dataset BeijingAir_Tiantan \
  --batch-size 64 \
  --n-test-samples 10000 \
  --eval-data-seed 1806770612 \
  --output-dir forecast_plots_final_traces/beijing_air \
  --sample PatchTST:drift:0:2329:572:0.596497 \
  --sample DLinear:spike:3:8609:3644:0.492074 \
  --export-traces

uv run python scripts/render_forecast_plots.py from-runs \
  --dataset Penmanshiel_Hourly_WT08 \
  --batch-size 64 \
  --n-test-samples 10000 \
  --eval-data-seed 1515776824 \
  --output-dir forecast_plots_final_traces/penmanshiel \
  --sample PatchTST:noise:2:4638:1136:0.243317 \
  --sample ModernTCN:attenuation:1:744:1661:0.884286 \
  --export-traces

uv run python scripts/render_forecast_plots.py from-runs \
  --dataset traffic \
  --batch-size 16 \
  --n-test-samples 10000 \
  --eval-data-seed 74851880 \
  --output-dir forecast_plots_final_traces/traffic \
  --sample PatchTST:time_stretch:4:7218:79:0.494280 \
  --sample PatchTST:missing_data:7:664:2366:0.491388 \
  --export-traces

uv run python scripts/render_forecast_plots.py from-runs \
  --dataset ETTh1 \
  --batch-size 16 \
  --n-test-samples 10000 \
  --eval-data-seed 341970080 \
  --output-dir forecast_plots_final_traces/etth1 \
  --sample PatchTST:time_compress:5:7968:2167:0.495240 \
  --sample PatchTST:stuck_sensor:6:8111:427:0.186316 \
  --export-traces
