# EchoGemma

Multimodal echocardiography report generation from DICOM studies. EchoGemma combines an EchoPrime video encoder and a LoRA-fine-tuned MedGemma language model to process full echocardiographic studies and generate clinical text reports.

# Setup

## 1. Install dependencies

```bash
uv sync
source .venv/bin/activate
```

## 2. Download the model
Note that you might need to log in to your Hugging Face account
```bash
python -m scripts.download
```

## 3. Download a sample study

```bash
wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/model_data.zip
unzip model_data.zip
mv model_data/example_study .
rm -rf model_data model_data.zip __MACOSX
```

## 4. Run inference

```bash
python -m scripts.inference_example
```
You'll get a report like this:
```
The user uploaded 48 videos, and the corresponding views are: A4C, Apical_Doppler, A4C, Doppler_Parasternal_Long, A4C, Apical_Doppler, Apical_Doppler, Parasternal_Long, A4C, Doppler_Parasternal_Short, Apical_Doppler, Parasternal_Long, Parasternal_Short, A4C, A3C, Parasternal_Short, Parasternal_Long, Apical_Doppler, Subcostal, A3C, Apical_Doppler, Apical_Doppler, Parasternal_Short, SSN, Subcostal, Apical_Doppler, Doppler_Parasternal_Short, A3C, Parasternal_Short, Doppler_Parasternal_Short, A2C, Apical_Doppler, A2C, A4C, Parasternal_Long, A3C, Parasternal_Short, A2C, Doppler_Parasternal_Short, Parasternal_Short, Parasternal_Long, Doppler_Parasternal_Long, Apical_Doppler, Subcostal, Apical_Doppler, Parasternal_Short, Apical_Doppler, SSN, A5C, Apical_Doppler.Aorta: There is mild aortic root dilation. Sinus of Valsalva: 3.9 cm. There is mild ascending aorta dilation. Ascending Aorta 3.9 cm. [SEP] Aortic Valve: Normal appearance and function of the aortic valve. No significant aortic stenosis or insufficiency. Trileaflet aortic valve. The peak transaortic gradient is 4 mmHg The mean transaortic gradient is 2 mmHg The aortic valve area by the continuity equation (using VTI) is 3.1 cm2 The aortic valve area by the continuity equation (using Vmax) is 3.1 cm2. No aortic regurgitation seen. [SEP] Atrial Septum: The interatrial septum is normal in appearance. [SEP] IVC: The inferior vena cava is of normal size. The IVC diameter is 16 mm The inferior vena cava shows a normal respiratory collapse consistent with normal right atrial pressure (3 mmHg). [SEP] Left Atrium: The left atrium is normal in size. [SEP] Left Ventricle: Normal left ventricular size by linear cavity dimension. Normal left ventricular size by volume Mild left ventricular hypertrophy. Normal left ventricular systolic function. LV Ejection Fraction is 55 %. Mild diastolic dysfunction. There is reversal of the E to A ratio and/or prolonged deceleration time consistent with impaired left ventricular relaxation. Doppler parameters and/or lateral mitral annular (E`) velocities are consistent with normal left ventricular filling pressures. [SEP] Mitral Valve: The mitral valve demonstrates normal leaflet morphology. The mitral valve demonstrates normal function with trace physiologic regurgitation. There is trivial mitral regurgitation. The peak transmitral gradient is 3 mmHg The mean transmitral gradient is 1 mmHg The mitral valve area by pressure half-time is 3.2 cm2. [SEP] Pericardium: Normal pericardium with no pericardial effusion. [SEP] Pulmonary Artery: Estimated PA Pressure is 18 mmHg. PA systolic pressure is normal. [SEP] Pulmonary Veins: Pulmonary veins are normal in appearance and pulse Doppler interrogation shows normal systolic predominant flow. [SEP] Pulmonic Valve: Normal pulmonic valve appearance. Normal pulmonic valve function with trace physiologic regurgitation. [SEP] Resting Segmental Wall Motion Analysis: Total wall motion score is 1.00. There are no regional wall motion abnormalities [SEP] Right Atrium: The right atrium is normal in size. [SEP] Right Ventricle: Normal right ventricular size. Normal right ventricular systolic function. [SEP] Tricuspid Valve: Normal appearance of the tricuspid valve. Est RV/RA pressure gradient is 15 mmHg. Estimated peak RVSP is 18 mmHg. There is trivial tricuspid regurgitation. [SEP]```