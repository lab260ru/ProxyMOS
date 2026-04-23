from datasets import load_dataset
import soundfile as sf
from pathlib import Path
import  pandas as pd
from tqdm import tqdm




outdir = Path("tts_our/audio")
outdir.mkdir(parents=True, exist_ok=True)


dataset = load_dataset("MTUCI/TTS_MOS_Evaluation_Dataset", split="train", token='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')

mos_records = []

for item in tqdm(dataset, desc="Сохраняем аудио"):
    safe_name = item['filename'].replace('/', '_').replace('\\', '_')
    out_path = outdir / safe_name
    out_path.write_bytes(item['audio'])
    mos_records.append({
        "inner_audio_id": item["inner_audio_id"],
        "file_name": "/home/maxim/MOS_research/" +str(out_path),
        "mos": item['mos_mean']
    })


csv_path = "tts_our/mos_labels.csv"
pd.DataFrame(mos_records).to_csv(csv_path, index=False)

print(f"Готово! Всего {len(mos_records)} аудио. MOS метки сохранены в {csv_path}")
