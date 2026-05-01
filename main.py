
from src.data.acquire import acquire
from src.preprocessing.pipeline import run_pipeline, get_output_columns
from src.utils.io import save_outputs
 
PLACE = "Portland, Oregon, USA"
OUTPUT_DIR = "data/processed"
 
df_raw = acquire(PLACE, save_path="data/raw/pois_portland_raw.csv")
df_processed = run_pipeline(df_raw)
df_out = get_output_columns(df_processed)
save_outputs(df_out, OUTPUT_DIR, name="pois_processed")
 