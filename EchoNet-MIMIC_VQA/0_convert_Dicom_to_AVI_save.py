import pandas as pd
import numpy as np
from pathlib import Path
import pydicom
from pydicom.pixel_data_handlers import convert_color_space
import torchvision
from tqdm import tqdm
import os


def change_dicom_color(dcm_path):

    ds = pydicom.dcmread(dcm_path)
    pixels = ds.pixel_array
    
    if ds.PhotometricInterpretation == 'MONOCHROME2':
        pixels = np.stack((pixels,)*3, axis=-1)
    elif ds.PhotometricInterpretation in ['YRB_FULL', 'YBR_FULL_422']:
        pixels = convert_color_space(pixels, ds.PhotometricInterpretation, 'RGB')
        if len(pixels.shape) < 4:
            ecg_mask = np.logical_and(pixels[:,:,1] > 200, pixels[:,:,0] < 100)
            pixels[ecg_mask,:] = 0
    elif ds.PhotometricInterpretation == 'RGB':
        if len(pixels.shape) < 4:
            ecg_mask = np.logical_and(pixels[:,:,1] > 200, pixels[:,:,0] < 100)
            pixels[ecg_mask,:] = 0
    else:
        print(f'Unsupported photometric interpretation: {ds.PhotometricInterpretation}')
    
    return pixels


def main():
    csv_path = '/YOUR workspace/..../MIMIC_Echo_1qa_SDE_vFINAL.csv'
    output_dir = Path('/YOUR workspace/.../ Echo_mp4') #directory to save mp4 files
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total records: {len(df)}")
    
    if 'dicom_path' not in df.columns:
        raise ValueError("CSV file must have 'dicom_path' column, Please enter absolute dicom file path")
    
    success_count = 0
    error_count = 0
    errors = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting DICOMs to MP4"):
        dicom_path = row['dicom_path']
        
        file_id = row['question_id']
        output_path = output_dir / f"Echo_{file_id}.mp4"
        
        if output_path.exists():
            continue
        
        try:
            if not os.path.exists(dicom_path):
                error_count += 1
                errors.append((idx, dicom_path, "File not found"))
                continue
            
            pixels = change_dicom_color(dicom_path)
            torchvision.io.write_video(str(output_path), pixels, fps=30)
            success_count += 1
            
        except Exception as e:
            error_count += 1
            errors.append((idx, dicom_path, str(e)))
    
    if errors:
        error_log_path = output_dir / 'conversion_errors.txt'
        with open(error_log_path, 'w') as f:
            f.write("Conversion Errors\n")
            f.write("="*50 + "\n\n")
            for idx, path, error in errors:
                f.write(f"Row {idx}: {path}\n")
                f.write(f"Error: {error}\n\n")
        print(f"\nError log saved to: {error_log_path}")
    
    print(f"\nMP4 files saved to: {output_dir}")


if __name__ == "__main__":
    main()
