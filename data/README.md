# Data Directory

This directory is for local data only. Large raw and processed data files should not be committed by default.

Suggested layout:

```text
data/raw/         Original trajectory datasets
data/processed/   Encoded images, resized images, and intermediate arrays
```

TODO: Document the expected schema for the trajectory file, including columns such as `vehicle_id`, `type`, `lon`, `lat`, and `speed`.
